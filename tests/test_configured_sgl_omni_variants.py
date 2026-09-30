"""Phase-3 converted server variants: configured payloads, menus and shared storage.

The expected payloads below were captured from the legacy per-server classes
immediately before their removal (phase 4); they pin the configured route to
the exact wire behavior the legacy implementations had.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from catalog_toml_support import read_catalog, write_catalog
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu import VoiceConfiguredSglOmniMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.sgl_omni_configured import SglOmniBackendAdapter, ConfiguredSettings
from tts_audiobook_tool.tts_models.model_catalog import parse_spec
from tts_audiobook_tool.tts_models.sgl_omni_definition import DEFINITION_PATH, load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

IDS = ("auk_sglomni", "auk_flash_sglomni", "fish_s2_sglomni", "higgs_v3_sglomni",
       "moss_delay_sglomni", "moss_local_sglomni", "qwen3tts_sglomni", "zonos2_sglomni")

# Nonstreaming expectations assume a 4s mocked reference (96000 zeros at
# 24kHz), speed 1.25 for AuK, French language, the second voice sample
# selected, and random seed 12345 where the variant sends a resolved seed.
GOLDEN_NONSTREAMING = {
    "auk_sglomni": [
        {"input": prompt, "stream": False, "response_format": "wav", "seed": 12345,
         "references": [{"audio_path": "data:{second}", "text": "éclair"}],
         "stage_params": {"auk_engine": {"gen_seconds": seconds}}}
        for prompt, seconds in (("One word", 3.657142857142857), ("Different words here", 9.142857142857142))
    ],
    "auk_flash_sglomni": None,  # same shape as auk_sglomni
    "fish_s2_sglomni": [
        {"input": prompt, "stream": False, "temperature": 0.7, "top_p": 0.9, "top_k": 30,
         "references": [{"audio_path": "data:{second}", "text": "éclair"}]}
        for prompt in ("One word", "Different words here")
    ],
    "higgs_v3_sglomni": [
        {"input": prompt, "stream": False, "temperature": 1.0, "top_p": 1.0, "top_k": 100,
         "max_tokens": 1536,
         "references": [{"audio_path": "data:{second}", "text": "éclair"}]}
        for prompt in ("One word", "Different words here")
    ],
    "moss_delay_sglomni": [
        {"input": prompt, "stream": False, "temperature": 1.7, "audio_top_p": 0.8,
         "audio_top_k": 25, "seed": 12345, "max_new_tokens": 1024, "language": "French",
         "references": [{"audio_path": "data:{second}", "text": "éclair"}]}
        for prompt in ("One word", "Different words here")
    ],
    "moss_local_sglomni": [
        {"input": prompt, "stream": False, "temperature": 1.0, "audio_top_p": 0.95,
         "audio_top_k": 50, "seed": 12345, "max_new_tokens": 1024, "language": "French",
         "references": [{"audio_path": "data:{second}", "text": "éclair"}]}
        for prompt in ("One word", "Different words here")
    ],
    "qwen3tts_sglomni": [
        {"input": prompt, "stream": False, "temperature": 0.9, "top_p": 1.0, "top_k": 50,
         "repetition_penalty": 1.3,
         "references": [{"audio_path": "data:{second}", "text": "éclair"}]}
        for prompt in ("One word", "Different words here")
    ],
    "zonos2_sglomni": [
        {"input": prompt, "stream": False, "max_new_tokens": tokens, "top_k": 100,
         "temperature": 1.15, "repetition_penalty": 1.2,
         "references": [{"audio_path": "data:{second}"}]}
        for prompt, tokens in (("One word", 280), ("Different words here", 320))
    ],
}

GOLDEN_STREAMING = {
    "fish_s2_sglomni": {"input": "hello", "stream": True, "temperature": 0.7, "top_p": 0.9,
                       "top_k": 30, "references": [{"audio_path": "data:{sample}", "text": "spoken"}]},
    "higgs_v3_sglomni": {"input": "hello", "stream": True, "temperature": 1.0, "top_p": 1.0,
                        "top_k": 100, "max_tokens": 1536,
                        "references": [{"audio_path": "data:{sample}", "text": "spoken"}]},
    "qwen3tts_sglomni": {"input": "hello", "stream": True, "temperature": 0.9, "top_p": 1.0,
                        "top_k": 50, "response_format": "pcm",
                        "references": [{"audio_path": "data:{sample}"}]},
    "zonos2_sglomni": {"input": "hello", "stream": True, "max_new_tokens": 240, "top_k": 100,
                      "temperature": 1.15, "repetition_penalty": 1.2,
                      "references": [{"audio_path": "data:{sample}"}], "response_format": "pcm"},
}


def resolve_golden(payloads, first, second, sample):
    """Fill golden audio_path placeholders with the test run's real paths."""
    resolved = json.loads(json.dumps(payloads))
    stack = [resolved] if isinstance(resolved, list) else [resolved]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str):
                    item[key] = value.format(first=first, second=second, sample=sample)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return resolved


@pytest.fixture
def server_mode(monkeypatch):
    old = (Tts._backend_mode, getattr(Tts, "_type", None),
           Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized,
           Tts._audio_cpp_definitions, Tts._selected_server_model_id, Tts._remote_issue)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsRuntimeMode.REMOTE_CLIENT))
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", classmethod(lambda cls: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.SGL_OMNI,
        candidates=((Tts.get_active_type(), Tts._selected_server_model_id),))))

    def install(model_id):
        Tts._configured_runtime = None
        Tts.init_local_model_type()
        Tts._type = TtsModelType.require_by_id(model_id)
        Tts._selected_server_model_id = model_id
        return Tts.get_instance()

    yield install
    TtsModelType.reset_catalog()
    REGISTRY.reset_to_builtins()
    (Tts._backend_mode, previous_type,
     Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime,
     Tts._catalog_initialized, Tts._audio_cpp_definitions, Tts._selected_server_model_id, Tts._remote_issue) = old
    if previous_type is None:
        if hasattr(Tts, "_type"):
            delattr(Tts, "_type")
    else:
        Tts._type = previous_type


@pytest.fixture
def capture(monkeypatch):
    calls = []
    sound = Sound(np.array([0.1], dtype=np.float32), 24000)
    monkeypatch.setattr(SglOmniUtil, "get_base_url", lambda: "http://example.test")
    monkeypatch.setattr(SglOmniUtil, "check_readiness", lambda _: pytest.fail("unexpected per-call readiness probe"))
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", lambda path: "data:" + path)
    monkeypatch.setattr(SoundFileUtil, "load", lambda path: Sound(np.zeros(96000, dtype=np.float32), 24000))
    monkeypatch.setattr("random.randrange", lambda *args: 12345)

    def concurrent(url, payloads, **kwargs):
        calls.extend(payloads)
        return [sound] * len(payloads)

    def streaming(url, payload, **kwargs):
        calls.append(payload)
        if kwargs["on_stream_chunk"]:
            kwargs["on_stream_chunk"](sound.data)
        if kwargs["on_stream_end"]:
            kwargs["on_stream_end"]()
        return sound

    monkeypatch.setattr(SglOmniUtil, "generate_concurrent", concurrent)
    monkeypatch.setattr(SglOmniUtil, "generate_streaming", streaming)
    return calls


def test_variant_switch_clears_previous_adapter_and_reuses_current(server_mode, monkeypatch):
    from tts_audiobook_tool.app_support import app_memory
    monkeypatch.setattr(app_memory, "gc_ram_vram", lambda: None)
    first = server_mode("fish_s2_sglomni")
    assert Tts.get_instance() is first
    Tts.set_type(TtsModelType.require_by_id("qwen3tts_sglomni"))
    second = Tts.get_instance()
    assert second is not first
    assert second is Tts.get_instance()
    assert second.INFO.id == "qwen3tts_sglomni"
    Tts.clear_tts_model()
    assert Tts.get_instance_if_exists() is None


def test_all_builtins_are_configured_without_catalog_duplication(server_mode):
    catalog = read_catalog(DEFINITION_PATH)
    assert catalog["schema_version"] == 1
    assert any(entry["id"] == "none" and "backend_kind" not in entry for entry in catalog["models"])
    assert any(entry.get("backend_kind") == "local" for entry in catalog["models"])
    assert all(entry["backend_kind"] == "sgl_omni"
               for entry in catalog["models"] if entry["id"] in IDS)
    assert all("symbol" not in entry for entry in catalog["models"])
    for entry in catalog["models"]:
        handle = TtsModelType.require_by_id(entry["id"])
        assert handle.id == entry["id"]
        assert TtsModelType.require_by_id(entry["id"]) is handle
        assert handle.value == parse_spec(entry)
    assert [handle.id for handle in TtsModelType.all()] == [entry["id"] for entry in catalog["models"]]
    definitions = load_definitions().models
    # Additional data-only shipped models are available through the same lookup.
    assert set(IDS) <= set(definitions)
    assert [id for id in definitions if id in IDS] == list(IDS)
    server_mode("higgs_v3_sglomni")
    assert all(Tts.get_configured_definition(TtsModelType.require_by_id(id)).spec == definitions[id].spec for id in IDS)
    assert [handle.id for handle in TtsModelType.all() if handle.id in IDS] == list(IDS)
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id("auk-flash") is TtsModelType.require_by_id("auk_flash_sglomni")
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id("moss-tts-local") is TtsModelType.require_by_id("moss_local_sglomni")
    assert len({TtsModelType.require_by_id(id) for id in IDS}) == len(IDS)


@pytest.mark.parametrize("model_id", IDS)
def test_configured_payload_matches_recorded_legacy_payload(model_id, server_mode, capture, tmp_path):
    for name in ("first.flac", "second.flac"):
        (tmp_path / name).write_bytes(b"audio")
    instance = server_mode(model_id)
    assert isinstance(instance, SglOmniBackendAdapter)
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
    voice_binding = REGISTRY.get(model_id, "file_name")
    project.set_model_setting(model_id, "file_name", ["first.flac", "second.flac"])
    if (model_id, "transcript") in REGISTRY.bindings:
        project.set_model_setting(model_id, "transcript", ["first", "éclair"])
    if model_id in ("auk_sglomni", "auk_flash_sglomni"):
        project.set_model_setting(model_id, "speed", 1.25)
    if model_id == "qwen3tts_sglomni":
        project.set_model_setting(model_id, "repetition_penalty", 1.3)
    project.language_code = "fr"
    prompts = ["One word", "Different words here"]
    assert isinstance(instance.generate_using_project(project, prompts, voice_selection_index=1), list)
    first, second = tmp_path / "first.flac", tmp_path / "second.flac"
    expected = resolve_golden(
        GOLDEN_NONSTREAMING["auk_sglomni"] if model_id == "auk_flash_sglomni" else GOLDEN_NONSTREAMING[model_id],
        first, second, second)
    assert capture == expected
    assert capture[0]["references"][0]["audio_path"].endswith("second.flac")
    assert project.get_model_setting(model_id, "file_name") == ["first.flac", "second.flac"]
    assert voice_binding.group == REGISTRY.get(model_id, "file_name").group
    if model_id in ("auk_sglomni", "auk_flash_sglomni", "moss_delay_sglomni", "moss_local_sglomni"):
        assert capture[0]["seed"] == 12345
    else:
        assert "seed" not in capture[0]


@pytest.mark.parametrize("model_id", ["fish_s2_sglomni", "higgs_v3_sglomni", "qwen3tts_sglomni", "zonos2_sglomni"])
def test_streamed_payload_and_callbacks_match_legacy(model_id, server_mode, capture, tmp_path):
    sample = tmp_path / "sample.flac"
    sample.write_bytes(b"audio")
    instance = server_mode(model_id)
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
    project.set_model_setting(model_id, "file_name", ["sample.flac"])
    if model_id in ("higgs_v3_sglomni", "fish_s2_sglomni"):
        project.set_model_setting(model_id, "transcript", ["spoken"])
    signals = []
    callbacks = {"on_stream_chunk": lambda _: signals.append("chunk"),
                 "on_stream_end": lambda: signals.append("end")}
    instance.generate_using_project(project, ["hello"], **callbacks)
    assert capture == [resolve_golden(GOLDEN_STREAMING[model_id], sample, sample, sample)]
    assert signals == ["chunk", "end"]
    assert capture[0]["stream"] is True
    if model_id in ("qwen3tts_sglomni", "zonos2_sglomni"):
        assert capture[0]["response_format"] == "pcm"
    else:
        assert "response_format" not in capture[0]


def test_fish_shared_override_cap_and_reset_preserve_local_value(server_mode, capture, monkeypatch):
    server_mode("fish_s2_sglomni")
    definition = Tts.get_configured_definition(Tts.get_active_type())
    project = Project(tts_model_type=Tts.get_active_type().id)
    monkeypatch.setattr(Project, "save", lambda self: "")
    project.set_model_setting("fish_s2_local", "top_k", 73)
    assert ConfiguredSettings.get(project, definition.parameters["top_k"], "fish_s2_sglomni") == 73
    assert any("clamp to 30" in warning for warning in Tts.get_model_support(project).get_warning_issues(project))
    Tts.get_instance().generate_using_project(project, ["hello"])
    assert capture[0]["top_k"] == 30
    assert project.get_model_setting("fish_s2_local", "top_k") == 73
    assert ProjectSerializationUtil.to_project_json_dict(project)["model_settings"]["shared"]["fish_s2"]["parameters"]["top_k"] == 73
    assert ConfiguredSettings.set(project, definition.parameters["top_p"], 0.8, "fish_s2_sglomni") == ""
    assert project.get_model_setting("fish_s2_local", "top_p") == 0.8
    assert ConfiguredSettings.set(project, definition.parameters["top_k"], None, "fish_s2_sglomni") == ""
    assert project.get_model_setting("fish_s2_local", "top_k") == -1
    assert "top_k" not in project.model_settings.shared["fish_s2"].get("parameters", {})


def test_auk_speed_utf8_and_seed_are_one_per_call(server_mode, capture, tmp_path):
    (tmp_path / "sample.flac").write_bytes(b"audio")
    instance = server_mode("auk_flash_sglomni")
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
    project.set_model_setting("auk_flash_sglomni", "file_name", ["sample.flac"])
    project.set_model_setting("auk_flash_sglomni", "transcript", ["é"])
    project.set_model_setting("auk_sglomni", "speed", 2.0)
    instance.generate_using_project(project, ["à", "abcd"], force_random_seed=True)
    assert [payload["stage_params"]["auk_engine"]["gen_seconds"] for payload in capture] == [2.0, 4.0]
    assert [payload["seed"] for payload in capture] == [12345, 12345]
    assert all("speed" not in payload and payload["response_format"] == "wav" for payload in capture)


@pytest.mark.parametrize("stored_seed, force_random, cap, expected, stop", [
    (-1, False, -1, 2**32 - 2, 2**32 - 1),
    (-1, False, 7, 7, 8),
    (-1, False, 0, 0, 1),
    (999, True, 7, 7, 8),
    (999, False, 7, 999, None),
])
def test_sgl_random_seed_cap_preserves_seed_reuse_and_fixed_values(
        server_mode, capture, monkeypatch, tmp_path, stored_seed, force_random, cap, expected, stop):
    (tmp_path / "sample.flac").write_bytes(b"audio")
    instance = server_mode("auk_flash_sglomni")
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
    project.set_model_setting("auk_flash_sglomni", "file_name", ["sample.flac"])
    project.set_model_setting("auk_flash_sglomni", "transcript", ["spoken"])
    project.set_model_setting("auk_flash_sglomni", "seed", stored_seed)
    draws = []

    def draw(start, end):
        draws.append((start, end))
        return end - 1

    monkeypatch.setattr("tts_audiobook_tool.tts_models.sgl_omni_configured.random.randrange", draw)
    result = instance.generate_using_project(
        project, ["first", "second"], force_random_seed=force_random, max_random_seed=cap)
    assert not isinstance(result, str)
    assert draws == ([] if stop is None else [(0, stop)])
    assert [payload["seed"] for payload in capture] == [expected, expected]
    assert all("max_random_seed" not in payload for payload in capture)


def test_seed_omitting_sgl_backend_ignores_random_cap(server_mode, capture, monkeypatch):
    instance = server_mode("zonos2_sglomni")
    monkeypatch.setattr("tts_audiobook_tool.tts_models.sgl_omni_configured.random.randrange",
                        lambda *_: pytest.fail("seed-omitting backend must not draw a seed"))
    result = instance.generate_using_project(
        Project(tts_model_type=Tts.get_active_type().id), ["hello"],
        force_random_seed=True, max_random_seed=0)
    assert not isinstance(result, str)
    assert "seed" not in capture[0] and "max_random_seed" not in capture[0]


def test_qwen_optional_penalty_and_optional_reference_text(server_mode, capture, tmp_path):
    (tmp_path / "sample.flac").write_bytes(b"audio")
    instance = server_mode("qwen3tts_sglomni")
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
    project.set_model_setting("qwen3tts_sglomni", "file_name", ["sample.flac"])
    instance.generate_using_project(project, ["hello"])
    assert "repetition_penalty" not in capture[0]
    assert "text" not in capture[0]["references"][0]
    assert "seed" not in capture[0] and "response_format" not in capture[0]
    project.set_model_setting("qwen3tts_local", "repetition_penalty", 1.05)
    capture.clear()
    instance.generate_using_project(project, ["hello"])
    assert capture[0]["repetition_penalty"] == 1.05


def test_zonos_per_prompt_tokens_and_no_transcript(server_mode, capture):
    instance = server_mode("zonos2_sglomni")
    instance.generate_using_project(Project(tts_model_type=Tts.get_active_type().id), ["word", " ".join(["word"] * 100)])
    assert [payload["max_new_tokens"] for payload in capture] == [240, 4096]
    assert all("seed" not in payload for payload in capture)


def test_moss_variants_language_music_and_no_rolling_block(server_mode, capture):
    for model_id, rate, local in (("moss_delay_sglomni", 24000, False), ("moss_local_sglomni", 48000, True)):
        server_mode(model_id)
        project = Project(tts_model_type=Tts.get_active_type().id)
        project.language_code = "fr"
        project.set_model_setting("moss_local", "batch_size", 3)
        project.set_model_setting("moss_local", "rolling_cont", 2)
        support = Tts.get_model_support(project)
        assert support.get_blocking_issues(project, None) == []
        assert support.get_warning_issues(project) == ["Using MOSS-TTS language value: French"]
        assert support.get_output_sample_rate(project) == rate
        assert support.can_hallucinate_music(project) is local
        assert support.should_trim_trailing_token_noise(project) is local


def test_moss_variants_do_not_inherit_local_rolling_blocker(server_mode, capture, tmp_path):
    (tmp_path / "sample.flac").write_bytes(b"audio")
    for model_id in ("moss_delay_sglomni", "moss_local_sglomni"):
        instance = server_mode(model_id)
        project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
        project.set_model_setting("moss_local", "file_name", ["sample.flac"])
        project.set_model_setting("moss_local", "batch_size", 3)
        project.set_model_setting("moss_local", "rolling_cont", 2)
        assert isinstance(instance.generate_using_project(project, ["hello"]), list)


def test_menu_seed_and_shared_fish_edit_without_storage_mutation(server_mode, monkeypatch):
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *_: [])
    for id, size in (("auk_sglomni", 2), ("moss_delay_sglomni", 4), ("fish_s2_sglomni", 3)):
        server_mode(id)
        items = VoiceConfiguredSglOmniMenu.make_items(SimpleNamespace(project=Project(tts_model_type=Tts.get_active_type().id)), Tts.get_configured_definition(Tts.get_active_type()))
        assert len(items) == size
        if id != "fish_s2_sglomni":
            assert "Seed" in items[-1].label


def test_readiness_selected_reference_and_auk_all_samples(server_mode, capture, tmp_path):
    (tmp_path / "first.flac").write_bytes(b"audio")
    for id in ("fish_s2_sglomni", "auk_sglomni"):
        instance = server_mode(id)
        project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
        project.set_model_setting(id, "file_name", ["first.flac", "missing.flac"])
        project.set_model_setting(id, "transcript", ["first", ""])
        issues = Tts.get_model_support(project).get_blocking_issues(project, None)
        assert any(issue.short == "voice clone transcript" for issue in issues)
        if id == "auk_sglomni":
            assert any(issue.short == "voice sample" for issue in issues)
        assert "transcript" in instance.generate_using_project(project, ["hello"], voice_selection_index=1)
        project.set_model_setting(id, "transcript", ["first", "second"])
        assert "missing.flac" in instance.generate_using_project(project, ["hello"], voice_selection_index=1)
        assert not capture


def test_fish_menu_reset_and_server_edit_reach_local_owner(server_mode, monkeypatch):
    instance = server_mode("fish_s2_sglomni")
    project = Project(tts_model_type=Tts.get_active_type().id)
    project.set_model_setting("fish_s2_local", "top_k", 73)
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *_: [])
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.printt", lambda *_: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.print_feedback", lambda *_: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_input",
                        lambda **_: next(responses))
    responses = iter(("-1", "25"))
    state = SimpleNamespace(project=project)
    item = VoiceConfiguredSglOmniMenu.make_items(state, instance.definition)[-1]
    item.handler(state, item)
    assert project.get_model_setting("fish_s2_local", "top_k") == -1
    item.handler(state, item)
    assert project.get_model_setting("fish_s2_local", "top_k") == 25
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["model_settings"]["shared"]["fish_s2"]["parameters"]["top_k"] == 25
    assert "top_k" not in saved["model_settings"]["models"].get("fish_s2_sglomni", {}).get("parameters", {})


def test_auk_reference_load_failure_is_forwarded(server_mode, capture, monkeypatch, tmp_path):
    (tmp_path / "sample.flac").write_bytes(b"audio")
    instance = server_mode("auk_sglomni")
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": Tts.get_active_type().id})
    project.set_model_setting("auk_sglomni", "file_name", ["sample.flac"])
    project.set_model_setting("auk_sglomni", "transcript", ["words"])
    project.set_model_setting("auk_sglomni", "speed", 1.5)
    monkeypatch.setattr(SoundFileUtil, "load", lambda _: "invalid reference")
    assert instance.generate_using_project(project, ["hello"]) == "invalid reference"
    assert not capture
    project.set_model_setting("auk_sglomni", "speed", 1.0)
    instance.generate_using_project(project, ["hello"])
    assert "stage_params" not in capture[0] and "speed" not in capture[0]


def test_shared_storage_round_trip_and_v2_migration(server_mode):
    project = Project.model_validate({
        "version": 2, "fish_s2_top_k": 82, "fish_s2_seed": -1,
        "fish_s2_voice_file_name": ["one.flac"], "fish_s2_voice_transcript": ["one"],
        "fish_s2_server_concurrent_requests": 3,
    })
    server_mode("fish_s2_sglomni")
    serialized = ProjectSerializationUtil.to_project_json_dict(project)
    serialized["model_settings"]["models"]["future_model"] = {"future": [1, 2]}
    serialized["model_settings"]["shared"]["future_group"] = {"model_ids": ["future_model"], "future": True}
    project = Project.model_validate(serialized)
    assert project.get_model_setting("fish_s2_local", "top_k") == 82
    assert project.get_model_setting("fish_s2_sglomni", "top_k") == 82
    assert project.get_model_setting("fish_s2_local", "seed") == -1
    assert project.get_model_setting("fish_s2_sglomni", "file_name") == ["one.flac"]
    assert project.get_model_setting("fish_s2_sglomni", "concurrent_requests") == 3
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["model_settings"]["models"]["future_model"] == {"future": [1, 2]}
    assert saved["model_settings"]["shared"]["future_group"]["future"] is True
    assert "fish_s2_top_k" not in saved
    assert saved["model_settings"]["shared"]["fish_s2"]["model_ids"] == ["fish_s2_local", "fish_s2_sglomni"]
    assert saved["model_settings"]["models"]["fish_s2_sglomni"]["orchestration"]["concurrent_requests"] == 3


def test_missing_builtin_definition_cannot_fall_back_to_legacy(tmp_path, server_mode, monkeypatch):
    server_mode("fish_s2_sglomni")
    data = read_catalog(DEFINITION_PATH)
    data["models"] = [entry for entry in data["models"] if entry["id"] != "fish_s2_sglomni"]
    data["setting_groups"]["fish_s2"].remove("fish_s2_sglomni")
    path = write_catalog(tmp_path / "missing.toml", data)
    with pytest.raises(ValueError, match="missing built-in.*fish_s2_sglomni"):
        load_definitions(path)
    monkeypatch.setattr("tts_audiobook_tool.tts_models.sgl_omni_definition.load_definitions", lambda: load_definitions(path))
    with pytest.raises(ValueError, match="missing built-in.*fish_s2_sglomni"):
        Tts.init_local_model_type()
    assert not Tts._configured_definitions
    assert Tts._configured_runtime is None


def test_loader_rejects_policy_and_owner_collisions_atomically(tmp_path):
    source = read_catalog(DEFINITION_PATH)
    for model_id, mutate, error in (
        ("fish_s2_sglomni", lambda e: e["sgl_omni"]["parameters"]["top_k"].update(project_attr="qwen3_top_k"), "top_k.*project_attr"),
        ("zonos2_sglomni", lambda e: e["sgl_omni"]["behavior"].update(prompt_policy="eval"), "prompt_policy"),
        ("auk_sglomni", lambda e: e["sgl_omni"]["parameters"]["speed"].update(default_sentinel=-1), r"speed\.(?:default_sentinel|sentinel)"),
        ("qwen3tts_sglomni", lambda e: e["sgl_omni"]["request_stream_defaults"].update(input="bad"), "request_stream_defaults.input"),
    ):
        data = copy.deepcopy(source)
        mutate(next(e for e in data["models"] if e["id"] == model_id))
        path = write_catalog(tmp_path / "bad.toml", data)
        before = TtsModelType.require_by_id("auk_sglomni").value
        with pytest.raises(ValueError, match=error):
            load_definitions(path)
        assert TtsModelType.require_by_id("auk_sglomni").value is before


@pytest.mark.parametrize("model_id", IDS)
def test_legacy_server_route_cannot_be_instantiated(model_id, server_mode):
    """Phase 4 removed the legacy classes; the configured adapter is the only route."""
    import importlib

    server_mode(model_id)
    model_type = TtsModelType.require_by_id(model_id)
    assert Tts.get_instance() is Tts.get_instance_if_exists()
    assert model_type not in Tts._MODEL_REGISTRY
    with pytest.raises(Exception, match="Not implemented"):
        Tts.get_class_for_type(model_type)
    legacy_modules = [
        "auk_server_model", "fish_s2_server_model", "higgs_v3_server_model",
        "moss_server_model", "qwen3_server_model", "zonos2_server_model",
        "auk_server_base_model", "fish_s2_server_base_model", "higgs_v3_server_base_model",
        "moss_server_base_model", "qwen3_server_base_model", "zonos2_server_base_model",
    ]
    for name in legacy_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"tts_audiobook_tool.tts_models.{name}")
