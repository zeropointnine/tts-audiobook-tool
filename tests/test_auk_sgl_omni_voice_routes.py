"""AuK / AuK-Flash (SGL-Omni): per-voice route chosen by whether the voice has a sample.

A voice with a sample is cloned (the server ignores `instructions` then, so
they are not sent); a sample-less voice is generated from its instructions as
a voice description, with `gen_seconds` from upstream AuK's ordinary-speaking
estimate (`auk_duration`), since the server requires an explicit duration
without reference audio. There is no app-side Mode.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared as menu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.auk_duration import estimate_instruct_tts_seconds
from tts_audiobook_tool.tts_models.sgl_omni_behavior_auk import AukSglOmniBehavior
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport, SglOmniBackendAdapter
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.tts_models.voice_variation_warning import get_voice_variation_warnings

AUK = "auk_sglomni"
FLASH = "auk_flash_sglomni"
TEXT = "The old lighthouse keeper climbed the stairs one last time, counting every step."


@pytest.fixture
def install(monkeypatch):
    previous = (Tts._backend_mode, getattr(Tts, "_type", None), Tts._config_fingerprint,
                Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized,
                Tts._selected_server_model_id)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.REMOTE_CLIENT))
    monkeypatch.setattr(menu.AudioMetaUtil, "get_audio_duration", lambda _: None)

    def install_model(model_id: str):
        Tts.init_local_model_type()
        model_type = TtsModelType.require_by_id(model_id)
        Tts._type = model_type
        Tts._selected_server_model_id = f"tencent/{model_id}"
        monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
            backend_kind=TtsBackendKind.SGL_OMNI, candidates=((model_type, f"tencent/{model_id}"),)))
        return Tts.get_configured_definition(model_type)

    try:
        yield install_model
    finally:
        TtsModelType.reset_catalog()
        REGISTRY.reset_to_builtins()
        (Tts._backend_mode, old_type, Tts._config_fingerprint,
         Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized,
         Tts._selected_server_model_id) = previous
        if old_type is None:
            if hasattr(Tts, "_type"):
                delattr(Tts, "_type")
        else:
            Tts._type = old_type


def _project(tmp_path, model_id: str, *voices: Voice) -> Project:
    for voice in voices:
        if voice.file_name:
            (tmp_path / voice.file_name).write_bytes(b"audio")
    return Project(dir_path=str(tmp_path), tts_model_type=model_id, language_code="en",
                   voice_references=list(voices))


def _send(monkeypatch, definition, project, prompts=(TEXT,), **kwargs) -> tuple[object, list[dict]]:
    sent: list[dict] = []
    sound = Sound(np.zeros(1, dtype=np.float32), 24000)
    monkeypatch.setattr(SglOmniUtil, "get_base_url", staticmethod(lambda: "http://example.test"))
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", staticmethod(lambda path: "data:reference"))
    # A 4s reference, for the speed-adjusted clone estimate
    monkeypatch.setattr(SoundFileUtil, "load", lambda path: Sound(np.zeros(96000, dtype=np.float32), 24000))
    monkeypatch.setattr("random.randrange", lambda *args: 12345)
    monkeypatch.setattr(SglOmniUtil, "generate_concurrent",
                        lambda url, payloads, **_: sent.extend(payloads) or [sound for _ in payloads])
    adapter = SglOmniBackendAdapter(definition, ConfiguredModelSupport(definition))
    return adapter.generate_using_project(project, list(prompts), **kwargs), sent


# --- duration estimate (upstream port) ---------------------------------------

def _frames(seconds: float) -> float:
    """Truncate to whole F5 frames (256 samples at 24 kHz), as upstream does."""
    return int(seconds * 24000 / 256) * 256 / 24000


def _accumulated(rate: float, byte_count: int) -> float:
    """Upstream sums per character in floats; mirror that so frame truncation
    agrees at exact frame boundaries (eg 80 x 0.0656 truncates to 491 frames)."""
    total = 0.0
    for _ in range(byte_count):
        total += rate
    return total


@pytest.mark.parametrize("text, language, expected", [
    # ASCII bytes at 0.0656 s/byte
    (TEXT, "en", _frames(_accumulated(0.0656, len(TEXT)))),
    # Under 10 bytes: upstream's short-text factor (/0.3)
    ("Yes.", "en", _frames(_accumulated(0.0656, 4) / 0.3)),
    # CJK (and the CJK punctuation between it, 3 bytes each) at 0.0803 s/byte
    ("今天天气很好，我们一起去公园散步吧。", "zh", _frames(_accumulated(0.0803 * 3, 18))),
])
def test_estimate_matches_upstream_baseline(text, language, expected):
    assert estimate_instruct_tts_seconds(text, language) == pytest.approx(expected)


def test_estimate_language_fallback_is_upstreams():
    # Script detection wins over the declared language...
    assert estimate_instruct_tts_seconds(TEXT, "zh") == estimate_instruct_tts_seconds(TEXT, "en")
    # ...which only matters for text with no CJK or English words; anything but
    # en/zh falls back to the Chinese rate, as upstream does.
    cyrillic = "Привет, как дела?"
    assert estimate_instruct_tts_seconds(cyrillic, "ru") == estimate_instruct_tts_seconds(cyrillic, "zh")
    assert estimate_instruct_tts_seconds(cyrillic, "en-US") < estimate_instruct_tts_seconds(cyrillic, "ru")


# --- capability ----------------------------------------------------------------

@pytest.mark.parametrize("model_id", [AUK, FLASH])
def test_auk_declares_instructions_for_sample_less_voices_only(install, tmp_path, model_id):
    definition = install(model_id)
    support = ConfiguredModelSupport(definition)
    project = _project(tmp_path, model_id, Voice())
    assert isinstance(support.behavior, AukSglOmniBehavior)
    assert definition.voice_instructions and not definition.spec.requires_voice
    assert support.can_instructions(project)
    assert not support.uses_instructions_with_sample(project)
    assert not support.expects_voice_instructions(project)  # blank uses the server's default description
    assert "instructions_message" in definition.spec.ui


def test_other_instruction_models_keep_instructions_with_a_sample(install, tmp_path):
    definition = install("cosyvoice3_sglomni")
    project = _project(tmp_path, "cosyvoice3_sglomni", Voice())
    assert ConfiguredModelSupport(definition).uses_instructions_with_sample(project)
    for model_id in ("breeze_tts_2_audiocpp", "omnivoice_local"):
        other = Project(tts_model_type=model_id)
        assert Tts.get_model_support(other).uses_instructions_with_sample(other)


# --- requests ------------------------------------------------------------------

@pytest.mark.parametrize("model_id", [AUK, FLASH])
def test_sample_less_voice_sends_description_and_estimated_duration(install, monkeypatch, tmp_path, model_id):
    definition = install(model_id)
    project = _project(tmp_path, model_id, Voice(instructions={model_id: "  A warm, relaxed female voice.  "}))
    result, sent = _send(monkeypatch, definition, project, prompts=(TEXT, "Yes."))
    assert not isinstance(result, str)
    assert sent == [{
        "response_format": "wav", "input": prompt, "stream": False, "seed": 12345,
        "instructions": "A warm, relaxed female voice.",
        "stage_params": {"auk_engine": {"gen_seconds": estimate_instruct_tts_seconds(prompt, "en")}},
    } for prompt in (TEXT, "Yes.")]


def test_sample_less_voice_without_instructions_uses_server_default(install, monkeypatch, tmp_path):
    definition = install(AUK)
    project = _project(tmp_path, AUK, Voice(instructions={AUK: "   "}))
    _, sent = _send(monkeypatch, definition, project)
    assert "instructions" not in sent[0] and "references" not in sent[0]
    assert sent[0]["stage_params"] == {"auk_engine": {"gen_seconds": estimate_instruct_tts_seconds(TEXT, "en")}}


def test_sample_less_duration_follows_speech_speed(install, monkeypatch, tmp_path):
    definition = install(AUK)
    project = _project(tmp_path, AUK, Voice())
    project.set_model_setting(AUK, "speed", 1.25)
    _, sent = _send(monkeypatch, definition, project)
    assert sent[0]["stage_params"]["auk_engine"]["gen_seconds"] == pytest.approx(
        estimate_instruct_tts_seconds(TEXT, "en") / 1.25)
    assert "speed" not in sent[0]


@pytest.mark.parametrize("speed", [1.0, 2.0])
def test_voice_with_a_sample_clones_and_drops_instructions(install, monkeypatch, tmp_path, speed):
    definition = install(AUK)
    project = _project(tmp_path, AUK, Voice("a.flac", "abcd", instructions={AUK: "A deep voice."}))
    project.set_model_setting(AUK, "speed", speed)
    _, sent = _send(monkeypatch, definition, project, prompts=("abcd",))
    assert sent[0]["references"] == [{"audio_path": "data:reference", "text": "abcd"}]
    assert "instructions" not in sent[0]
    if speed == 1.0:
        assert "stage_params" not in sent[0]  # the server derives it from the sample
    else:
        # Unchanged clone estimate: 4s reference x 4/4 bytes / speed 2
        assert sent[0]["stage_params"] == {"auk_engine": {"gen_seconds": 2.0}}
    # Kept for when the sample is cleared
    assert project.voice_references[0].instructions == {AUK: "A deep voice."}


def test_routes_mix_per_voice_and_instructions_are_per_model(install, monkeypatch, tmp_path):
    definition = install(FLASH)
    project = _project(tmp_path, FLASH,
                       Voice("a.flac", "words", instructions={FLASH: "Unused with a sample."}),
                       Voice(instructions={AUK: "AuK only.", FLASH: "Flash description."}))
    _, cloned = _send(monkeypatch, definition, project, voice_selection_index=0)
    _, described = _send(monkeypatch, definition, project, voice_selection_index=1)
    assert "references" in cloned[0] and "instructions" not in cloned[0]
    assert "references" not in described[0] and described[0]["instructions"] == "Flash description."


def test_sample_less_voice_passes_preflight_with_a_variation_note(install, tmp_path):
    definition = install(AUK)
    project = _project(tmp_path, AUK, Voice(instructions={AUK: "A calm narrator."}))
    state = SimpleNamespace(project=project, prefs=SimpleNamespace(stt_variant=None, stt_config={}))
    support = ConfiguredModelSupport(definition)
    assert VoiceMenuShared.get_voice_problems(state) == []
    assert [issue for issue in support.get_blocking_issues(project) if issue.short != "SGL-Omni server"] == []
    assert get_voice_variation_warnings(project, [0]) == [
        "Generated voices may vary between segments "
        "(AuK instruction sets the kind of voice, but not a fixed speaker). (Voice 1)"]


@pytest.mark.parametrize("model_id, name", [(AUK, "AuK"), (FLASH, "AuK-Flash")])
def test_variation_note_is_auk_specific_only_for_an_in_play_described_voice(install, tmp_path, model_id, name):
    install(model_id)
    project = _project(tmp_path, model_id,
                       Voice("a.flac", "words", instructions={model_id: "Unused with a sample."}),
                       Voice(instructions={model_id: "A calm narrator."}),
                       Voice())
    specific = (f"Generated voices may vary between segments "
                f"({name} instruction sets the kind of voice, but not a fixed speaker).")
    generic = "Generated voices may vary because no voice reference has been configured."
    assert get_voice_variation_warnings(project, [0]) == []  # cloned: no note
    assert get_voice_variation_warnings(project, [1]) == [f"{specific} (Voice 2)"]
    assert get_voice_variation_warnings(project, [2]) == [f"{generic} (Voice 3)"]  # server's default description
    # A mixed run gets each note for its own voices
    assert get_voice_variation_warnings(project, [0, 1, 2]) == [f"{specific} (Voice 2)", f"{generic} (Voice 3)"]
    # Another model's instructions on the voice do not count
    project.voice_references[2].instructions["auk_flash_sglomni" if model_id == AUK else AUK] = "Other model."
    assert get_voice_variation_warnings(project, [2]) == [f"{generic} (Voice 3)"]


def test_voice_with_a_sample_still_requires_its_transcript(install, monkeypatch, tmp_path):
    definition = install(AUK)
    project = _project(tmp_path, AUK, Voice("a.flac", "", instructions={AUK: "A calm narrator."}))
    result, sent = _send(monkeypatch, definition, project)
    assert result == "Voice clone transcript required for every AuK voice sample" and not sent


# --- menus ---------------------------------------------------------------------

def _plain(state, value) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", get_string_from(state, value))


def _submenu_instructions_item(monkeypatch, project):
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.voice_sample_submenu(state, project.get_tts_model_type(), 0)
    items = captured["items"](state)
    return state, next(item for item in items if _plain(state, item.label).startswith("Instructions/AuK"))


@pytest.mark.parametrize("has_sample", [False, True])
@pytest.mark.parametrize("instructions", ["A warm voice.", "", "   "])
def test_instructions_item_is_always_editable_and_marked_unused_only_when_set(
        install, monkeypatch, tmp_path, has_sample, instructions):
    install(AUK)
    project = _project(tmp_path, AUK, Voice("a.flac" if has_sample else "", "words",
                                            instructions={AUK: instructions}))
    state, item = _submenu_instructions_item(monkeypatch, project)
    value = f'"{instructions}"' if instructions.strip() else "none"
    assert _plain(state, item.label) == f"Instructions/AuK (currently: {value})"
    # The proviso only applies to a value that a voice sample overrides
    if has_sample and instructions.strip():
        assert item.sublabel == menu.SUBLABEL_INSTRUCTIONS_UNUSED_WITH_SAMPLE
    else:
        assert item.sublabel is None


def test_voice_list_previews_instructions_only_when_used(install, tmp_path):
    install(AUK)
    project = _project(tmp_path, AUK, Voice("a.flac", "words", instructions={AUK: "Unused description."}),
                       Voice(instructions={AUK: "Used description."}), Voice())
    tts_type = project.get_tts_model_type()
    cloned, described, default = (
        re.sub(r"\x1b\[[0-9;]*m", "", VoiceMenuShared.make_voice_sample_item_label(project, tts_type, index)[0])
        for index in range(3))
    assert "a.flac" in cloned and "Unused description." not in cloned
    assert described == 'Voice 2: "Used description."'
    assert default == "Voice 3: no sample"  # not "(required)"
