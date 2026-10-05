"""Offline contracts for both audio.cpp MOSS architectures; no model inference."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from catalog_toml_support import read_catalog
from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.audio_cpp_configured import (
    AudioCppBackendAdapter, AudioCppModelSupport, AudioCppSettings,
)
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.audio_cpp_detection import detect_audio_cpp_models
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, parse_audio_cpp_declaration
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


VARIANTS = (
    ("moss_delay_audiocpp", "delay", "moss_tts_v15", 24000, (1.5, 0.6, 50), "MossTTSDelay"),
    ("moss_local_audiocpp", "local", "moss_tts_local", 48000, (1.7, 0.8, 25), "MossTTSLocal"),
)
IDS = tuple(item[0] for item in VARIANTS)


def definition(model_id: str):
    return load_audio_cpp_definitions().models[model_id]


def make_adapter(model_id: str):
    item = definition(model_id)
    return AudioCppBackendAdapter(item, AudioCppModelSupport(item), "opaque-operator-name")


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def generate(url, payload, print_request=False):
        calls.append(payload)
        return SimpleNamespace(data=np.asarray([len(calls)], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", generate)
    return calls


@pytest.mark.parametrize("model_id,prefix,family,rate,defaults,arch", VARIANTS)
def test_definitions_use_native_defaults_and_existing_moss_ui(model_id, prefix, family, rate, defaults, arch):
    item = definition(model_id)
    other = TtsModelType.require_by_id(f"moss_{prefix}_sglomni").value
    assert item.family == family and item.tasks == ("tts", "clon") and item.mode == "offline"
    assert item.session_options == {}
    assert item.request_options == ({"max_tokens": 1024} if prefix == "local" else {})
    assert item.spec.backend_kind is TtsBackendKind.AUDIO_CPP
    assert item.spec.default_output_sample_rate == rate
    assert not item.voice_required and not item.spec.requires_voice and not item.spec.can_stream
    assert not item.reference_transcript
    assert item.spec.ui["proper_name"] == other.ui["proper_name"]
    assert item.spec.ui["short_name"] == other.ui["short_name"]
    assert item.spec.substitutions == other.substitutions
    assert item.spec.file_tag == "moss"
    assert item.language_policy == "moss"
    assert item.language_target == ("options" if prefix == "delay" else "top_level")
    assert item.music_and_trim is (prefix == "local")
    expected = {
        f"{prefix}_temperature": ("float", defaults[0], 0.8, 3.0, "temperature"),
        f"{prefix}_top_p": ("float", defaults[1], 0.5, 1.0, "top_p"),
        f"{prefix}_top_k": ("int", defaults[2], 10, 100, "top_k"),
    }
    assert {
        name: (p.type, p.default, p.min, p.max, p.request_key)
        for name, p in item.parameters.items()
    } == expected
    assert all(p.target == "top_level" and p.default_sentinel == -1 for p in item.parameters.values())
    assert [(c.kind, c.parameter, c.label) for c in item.menu] == [
        ("voice_samples", "", ""),
        ("parameter", f"{prefix}_temperature", "Temperature"),
        ("parameter", f"{prefix}_top_p", "Top-P"),
        ("parameter", f"{prefix}_top_k", "Top-K"),
        ("seed", "", ""),
    ]
    assert REGISTRY.voice_binding(model_id).group == ""
    assert REGISTRY.transcript_binding(model_id) is None
    assert REGISTRY.orchestration_binding(model_id) is None
    assert not TtsModelType.require_by_id(model_id).can_batch()
    assert {b.name for b in REGISTRY.for_model(model_id)} == {*expected, "file_name", "seed"}
    project = Project(tts_model_type=model_id)
    for name, parameter in item.parameters.items():
        assert project.get_model_setting(model_id, name) == -1
        assert AudioCppSettings.get(project, parameter) == parameter.default


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("suffix,valid,invalid", [
    ("temperature", 0.8, 0.79), ("temperature", 3.0, 3.01),
    ("top_p", 0.5, 0.49), ("top_p", 1.0, 1.01),
    ("top_k", 10, 9), ("top_k", 100, 101), ("top_k", 25, 25.5),
    ("temperature", 1.0, float("nan")), ("top_p", 0.8, float("inf")),
    ("top_k", 50, True),
])
def test_parameter_validation_and_default_reset(monkeypatch, model_id, suffix, valid, invalid):
    monkeypatch.setattr(Project, "save", lambda self: "")
    prefix = "delay" if model_id == IDS[0] else "local"
    parameter = definition(model_id).parameters[f"{prefix}_{suffix}"]
    project = Project(tts_model_type=model_id)
    assert AudioCppSettings.set(project, parameter, valid) == ""
    assert AudioCppSettings.get(project, parameter) == valid
    with pytest.raises(ValueError):
        AudioCppSettings.set(project, parameter, invalid)
    assert AudioCppSettings.get(project, parameter) == valid
    assert AudioCppSettings.set(project, parameter, None) == ""
    assert project.get_model_setting(model_id, parameter.name) == -1
    assert AudioCppSettings.get(project, parameter) == parameter.default


@pytest.mark.parametrize("model_id,prefix,family,rate,defaults,arch", VARIANTS)
@pytest.mark.parametrize("overrides", [False, True])
def test_payload_defaults_overrides_language_and_sequential_order(
        capture, model_id, prefix, family, rate, defaults, arch, overrides):
    project = Project(tts_model_type=model_id, language_code="fr")
    project.set_model_setting(model_id, "seed", 2**32 - 1)
    values = (2.0, 0.75, 42) if overrides else defaults
    if overrides:
        for name, value in zip(("temperature", "top_p", "top_k"), values):
            project.set_model_setting(model_id, f"{prefix}_{name}", value)
    adapter = make_adapter(model_id)
    result = adapter.generate_using_project(project, ["first", "second"])
    assert not isinstance(result, str) and [sound.data[0] for sound in result] == [1, 2]
    assert [payload["input"] for payload in capture] == ["first", "second"]
    for prompt, payload in zip(("first", "second"), capture):
        expected = {
            "model": "opaque-operator-name", "input": prompt, "response_format": "wav",
            "seed": 2**32 - 1, "temperature": values[0], "top_p": values[1], "top_k": values[2],
        }
        if prefix == "delay":
            expected["options"] = {"language": "French"}
        else:
            expected["language"] = "French"
            expected["options"] = {"max_tokens": 1024}
        assert payload == expected  # Only Local pins a generation limit; no transcript/repetition/chunk controls.
    assert adapter.generate_using_project(project, [], on_stream_end=None) == []
    assert adapter.generate_using_project(project, ["stream"], on_stream_end=lambda: None).endswith(
        "does not support streaming")
    assert len(capture) == 2


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("language", ["", "xx"])
def test_unknown_language_is_omitted_from_both_payload_locations(capture, model_id, language):
    project = Project(tts_model_type=model_id, language_code=language)
    project.set_model_setting(model_id, "seed", 0)
    assert not isinstance(make_adapter(model_id).generate_using_project(project, ["hello"]), str)
    assert "language" not in capture[0]
    assert "language" not in capture[0].get("options", {})
    if model_id == IDS[1]:
        assert capture[0]["options"] == {"max_tokens": 1024}
    else:
        assert "options" not in capture[0]
    assert "Using MOSS-TTS language value: None" in AudioCppModelSupport(
        definition(model_id)).get_warning_issues(project)


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("transcript", ["", "ignored reference transcript"])
def test_selected_secondary_voice_works_without_transcript_and_never_transmits_one(
        monkeypatch, capture, tmp_path, model_id, transcript):
    project = Project(tts_model_type=model_id, dir_path=str(tmp_path), language_code="en")
    project.set_model_setting(model_id, "file_name", ["one.flac", "two.flac"])
    project.set_model_setting(model_id, "seed", 0)
    seen = []

    def reference(path):
        seen.append(Path(path).name)
        return "data:audio/wav;base64,mocked"

    monkeypatch.setattr(AudioCppUtil, "make_voice_ref", reference)
    # Even an external caller supplying a transcript cannot activate an unused field.
    current_pair = ProjectVoiceUtil.current_voice_reference_pair
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda p, t, index: (current_pair(p, t, index)[0], transcript))
    result = make_adapter(model_id).generate_using_project(project, ["hello"], voice_selection_index=1)
    assert not isinstance(result, str) and seen == ["two.flac"]
    assert capture[0]["voice_ref"] == {"type": "base64", "data": "data:audio/wav;base64,mocked"}
    assert "reference_text" not in capture[0] and "reference_text" not in capture[0].get("options", {})


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("stored_seed,force_random,caller_cap,expected,draw_stop", [
    (-1, False, -1, 2**32 - 1, 2**32),
    (123, True, 7, 7, 8), (123, False, 7, 123, None),
    (-1, False, 0, 0, 1),
])
def test_seed_uses_existing_uint32_random_fixed_and_caller_cap(
        monkeypatch, capture, model_id, stored_seed, force_random, caller_cap, expected, draw_stop):
    draws = []

    def draw(stop):
        draws.append(stop)
        return stop - 1

    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", draw)
    project = Project(tts_model_type=model_id)
    project.set_model_setting(model_id, "seed", stored_seed)
    assert not isinstance(make_adapter(model_id).generate_using_project(
        project, ["hello"], force_random_seed=force_random, max_random_seed=caller_cap), str)
    assert capture[0]["seed"] == expected
    assert draws == ([] if draw_stop is None else [draw_stop])


@pytest.mark.parametrize("stored_seed,force_random,caller_cap,expected,draw_stop", [
    (-1, False, -1, 2147483647, 2147483648),
    (-1, False, 2**32 - 1, 2147483647, 2147483648),
    (2**32 - 1, True, -1, 2147483647, 2147483648),
    (-1, False, 7, 7, 8),
    (-1, False, 0, 0, 1),
    (2147483647, False, -1, 2147483647, None),
    (123, False, 7, 123, None),
])
def test_delay_random_seed_int32_cap_reaches_payload_through_dispatch(
        monkeypatch, capture, stored_seed, force_random, caller_cap, expected, draw_stop):
    from tts_audiobook_tool.tts import Tts

    model_id = IDS[0]
    project = Project(tts_model_type=model_id)
    project.set_model_setting(model_id, "seed", stored_seed)
    adapter = make_adapter(model_id)
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id(model_id))
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: adapter))
    draws = []

    def draw(stop):
        draws.append(stop)
        return stop - 1  # Exercise the inclusive maximum that previously broke std::stoi.

    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", draw)
    result = Tts.generate_using_project(
        project, ["first", "second"], force_random_seed=force_random, max_random_seed=caller_cap)
    assert not isinstance(result, str)
    assert draws == ([] if draw_stop is None else [draw_stop])
    assert [payload["seed"] for payload in capture] == [expected, expected]
    assert all(0 <= payload["seed"] <= 2147483647 for payload in capture)
    assert all("max_random_seed" not in payload for payload in capture)
    assert project.get_model_setting(model_id, "seed") == stored_seed


@pytest.mark.parametrize("model_id", IDS)
def test_readiness_optional_voice_and_local_only_postprocessing(monkeypatch, tmp_path, model_id):
    project = Project(tts_model_type=model_id, dir_path=str(tmp_path), language_code="en")
    support = AudioCppModelSupport(definition(model_id))
    local_issues = lambda: [issue for issue in support.get_blocking_issues(project)
                           if issue.short != "audio.cpp server"]
    assert local_issues() == []
    assert support.get_voice_display_info(project).value.endswith("none")
    assert "Using MOSS-TTS language value: English" in support.get_warning_issues(project)
    assert support.can_hallucinate_music(project) is (model_id == IDS[1])
    assert support.should_trim_trailing_token_noise(project) is (model_id == IDS[1])
    project.set_model_setting(model_id, "file_name", ["missing.flac"])
    assert any("not found" in issue.verbose for issue in local_issues())
    (tmp_path / "missing.flac").write_bytes(b"reference")
    assert local_issues() == []  # No unused transcript requirement.
    project.set_model_setting("moss_local", "rolling_cont", 3)
    assert local_issues() == []  # No inherited local rolling-continuation blocker.


@pytest.mark.parametrize("model_id,prefix,family,rate,defaults,arch", VARIANTS)
def test_menu_has_only_voice_sampling_and_seed_controls(
        monkeypatch, model_id, prefix, family, rate, defaults, arch):
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items",
                        lambda *_, **__: [MenuItem("Add/remove voice samples", lambda *_: None)])
    monkeypatch.setattr(VoiceMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("Seed (currently: random)", lambda *_: None))
    state = SimpleNamespace(project=Project(tts_model_type=model_id))
    items = VoiceAudioCppMenu.make_items(state, definition(model_id))
    assert [text_util.strip_ansi_codes(get_string_from(state, item.label)) for item in items] == [
        "Add/remove voice samples",
        f"Temperature (currently: {defaults[0]:.2f} default)",
        f"Top-P (currently: {defaults[1]:.2f} default)",
        f"Top-K (currently: {defaults[2]} default)",
        "Seed (currently: random)",
    ]


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("response,expected,error", [
    ("-1", 2.3, "Out of range"), ("0.79", 2.3, "Out of range"),
    ("nan", 2.3, "Out of range"), ("inf", 2.3, "Out of range"),
    ("invalid", 2.3, "Bad value"), ("0.8", 0.8, ""), ("3.0", 3.0, ""),
    ("", 2.3, ""), ("2.3", 2.3, ""),
])
def test_numeric_menu_uses_standard_prompt_validation_and_save(
        monkeypatch, model_id, response, expected, error):
    saves = []
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))
    prefix = "delay" if model_id == IDS[0] else "local"
    parameter = definition(model_id).parameters[f"{prefix}_temperature"]
    project = Project(tts_model_type=model_id)
    project.set_model_setting(model_id, parameter.name, 2.3)
    printed, errors, prefills = [], [], []
    monkeypatch.setattr("tts_audiobook_tool.ask.printt", printed.append)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_error", errors.append)

    def answer(*, prefill):
        prefills.append(prefill)
        return response

    monkeypatch.setattr("tts_audiobook_tool.ask.ask_input", answer)
    state = SimpleNamespace(project=project)
    label = f"MossTTS{prefix.title()} temperature"
    control = VoiceAudioCppMenu.make_parameter_item(state, parameter, label)
    control.handler(state, control)
    assert text_util.strip_ansi_codes(printed[0]) == (
        f"Enter {label}: (valid range: 0.8-3.0; default: {parameter.default})")
    assert prefills == ["2.3"]
    assert errors == ([error] if error else [])
    assert saves == ([] if expected == 2.3 else [project])
    assert project.get_model_setting(model_id, parameter.name) == expected
    assert AudioCppSettings.get(project, parameter) == expected


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("suffix", ["temperature", "top_p", "top_k"])
def test_numeric_menu_prefills_native_default_without_reset_input_or_saving(
        monkeypatch, model_id, suffix):
    saves, prefills = [], []
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))
    monkeypatch.setattr("tts_audiobook_tool.ask.printt", lambda *_: None)
    prefix = "delay" if model_id == IDS[0] else "local"
    parameter = definition(model_id).parameters[f"{prefix}_{suffix}"]
    project = Project(tts_model_type=model_id)

    def answer(*, prefill):
        prefills.append(prefill)
        return prefill

    monkeypatch.setattr("tts_audiobook_tool.ask.ask_input", answer)
    state = SimpleNamespace(project=project)
    control = VoiceAudioCppMenu.make_parameter_item(state, parameter, "Value")
    control.handler(state, control)
    assert float(prefills[0]) == parameter.default
    assert project.get_model_setting(model_id, parameter.name) == -1
    assert saves == []


def test_sampling_and_voices_remain_private_and_roundtrip_without_pinning_defaults():
    project = Project(tts_model_type=IDS[0])
    project.set_model_setting(IDS[0], "delay_temperature", 2.0)
    project.set_model_setting(IDS[0], "file_name", ["delay.flac"])
    project.set_model_setting(IDS[0], "seed", 12)
    project.set_model_setting(IDS[1], "local_top_k", 70)
    project.set_model_setting(IDS[1], "file_name", ["local.flac"])
    project.set_model_setting("moss_local", "delay_temperature", 2.7)
    project.set_model_setting("moss_delay_sglomni", "delay_temperature", 2.5)
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert "moss" not in payload["model_settings"]["shared"]
    models = payload["model_settings"]["models"]
    assert models[IDS[0]]["parameters"] == {
        "delay_temperature": 2.0, "delay_top_p": None, "delay_top_k": None, "seed": 12,
    }
    assert models[IDS[1]]["parameters"] == {
        "local_temperature": None, "local_top_p": None, "local_top_k": 70, "seed": None,
    }
    for owner, prefix in zip(IDS, ("delay", "local")):
        assert "orchestration" not in models[owner]
        assert project.get_model_setting(owner, f"{prefix}_top_p") == -1
    restored = Project.model_validate(payload)
    for owner in (*IDS, "moss_local", "moss_delay_sglomni"):
        for binding in REGISTRY.for_model(owner):
            assert restored.get_model_setting(owner, binding.name) == project.get_model_setting(owner, binding.name)
    assert restored.get_model_setting(IDS[0], "file_name") == ["delay.flac"]
    assert restored.get_model_setting(IDS[1], "file_name") == ["local.flac"]
    for owner, prefix in zip(IDS, ("delay", "local")):
        parameter = definition(owner).parameters[f"{prefix}_top_p"]
        assert AudioCppSettings.get(restored, parameter) == parameter.default


@pytest.mark.parametrize("model_id,prefix,family,rate,defaults,arch", VARIANTS)
def test_detection_accepts_both_tasks_without_guessing_from_ids(model_id, prefix, family, rate, defaults, arch):
    handle = TtsModelType.require_by_id(model_id)
    models = [{"id": f"operator-{task}", "family": family, "task": task, "mode": "offline"}
              for task in ("tts", "clon")]
    assert detect_audio_cpp_models(models) == [(handle, model["id"]) for model in models]
    for change in ({"mode": "streaming"}, {"task": "clone"}, {"task": "asr"},
                   {"family": "moss_tts_nano"}, {"family": "moss_voicegen"}, {"family": "unknown"}):
        assert detect_audio_cpp_models([{**models[0], "id": "MOSS-TTS-v1.5", **change}]) == []


@pytest.mark.parametrize("field,value", [
    ("language_policy", "unknown"), ("language_policy", True),
    ("language_target", "unknown"), ("language_target", None),
    ("music_and_trim", "true"), ("music_and_trim", 1),
])
def test_new_schema_fields_reject_bad_types_and_values(field, value):
    catalog = read_catalog(CATALOG_PATH)
    entry = deepcopy(next(item for item in catalog["models"] if item["id"] == IDS[0]))
    entry["audio_cpp"][field] = value
    with pytest.raises(ValueError, match=field):
        parse_audio_cpp_declaration(entry, "test")


def test_existing_families_keep_default_language_policy_and_postprocessing():
    for model_id, item in load_audio_cpp_definitions().models.items():
        if model_id in IDS:
            continue
        assert item.language_policy == "project_code" and item.language_target == "top_level"
        assert not item.music_and_trim
        support = AudioCppModelSupport(item)
        assert not support.can_hallucinate_music(Project(tts_model_type=model_id))
        assert not support.should_trim_trailing_token_noise(Project(tts_model_type=model_id))
