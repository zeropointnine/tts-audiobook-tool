"""Offline Qwen3 audio.cpp checkpoint/mode contracts and consumer agreement."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.server.server import is_missing_required_transcript
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport, AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.audio_cpp_detection import detect_audio_cpp_models
from tts_audiobook_tool.tts_models.model_support import make_language_value_note
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

MODEL_ID = "qwen3tts_audiocpp"


@pytest.fixture
def definition():
    return load_audio_cpp_definitions().models[MODEL_ID]


def make_project(tmp_path, model_type="base", transcript="spoken words"):
    project = Project(dir_path=str(tmp_path), tts_model_type=MODEL_ID, language_code="en-US",
                      voice_references=[{"file_name": "sample.flac", "transcript": transcript,
                                         "presets": {MODEL_ID: "vivian"},
                                         "instructions": {MODEL_ID: "Speak warmly."}}])
    project.set_model_setting(MODEL_ID, "model_type", model_type)
    project.set_model_setting(MODEL_ID, "seed", 7)
    return project


def capture_requests(monkeypatch):
    sent = []
    monkeypatch.setattr(AudioCppUtil, "generate", lambda url, payload, **kwargs: sent.append(payload)
                        or SimpleNamespace(data=np.zeros(1, dtype=np.float32), sr=24000))
    return sent


def test_model_type_menu_subheading(monkeypatch, definition):
    # Catalog guidance reaches the generic option submenu, not a Qwen-specific renderer.
    from tts_audiobook_tool.menus.menu_util import MenuUtil

    control = next(control for control in definition.menu if control.parameter == "model_type")
    assert control.subheading == "Model type must match that of the audio.cpp's loaded checkpoint"
    state = SimpleNamespace(project=Project(tts_model_type=MODEL_ID))
    captured = {}
    monkeypatch.setattr(MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))
    item, = ModelAudioCppMenu.make_control_items(state, definition, control)
    item.handler(state, item)
    assert captured["subheading"] == control.subheading
    assert captured["values"] == ["base", "custom_voice", "voice_design"]


def test_defaults_and_choice_contracts(definition):
    project = Project(tts_model_type=MODEL_ID)
    assert AudioCppModelSupport(definition).get_values(project) == {
        "model_type": "base", "temperature": .9, "top_k": 50,
        "top_p": 1., "repetition_penalty": 1.05,
    }
    assert project.get_model_setting(MODEL_ID, "seed") == -1
    assert [c.value for c in definition.parameters["model_type"].choices] == ["base", "custom_voice", "voice_design"]
    assert "clone_mode" not in definition.parameters
    assert all(control.parameter != "clone_mode" for control in definition.menu)
    assert definition.family == "qwen3_tts" and definition.tasks == ("tts", "clon", "vdes")
    assert definition.mode == "offline" and definition.spec.default_output_sample_rate == 24000


@pytest.mark.parametrize("name,minimum,maximum", [
    ("temperature", .01, 3.), ("top_k", 1, 1000), ("top_p", .01, 1.),
    ("repetition_penalty", 1., 2.),
])
def test_numeric_bounds_and_types(definition, name, minimum, maximum):
    parameter = definition.parameters[name]
    assert (parameter.min, parameter.max) == (minimum, maximum)
    for value in (minimum, maximum):
        assert parameter.validate(value) == value
    for value in (True, minimum - 1, maximum + 1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            parameter.validate(value)
    if parameter.type == "int":
        with pytest.raises(ValueError):
            parameter.validate(1.5)


@pytest.mark.parametrize("value", ["Base", "unknown", True])
def test_model_type_rejects_unknown_values(definition, value):
    with pytest.raises(ValueError):
        AudioCppSettings.get(SimpleNamespace(get_model_setting=lambda *_: value), definition.parameters["model_type"])


@pytest.mark.parametrize("model_type", ["base", "custom_voice", "voice_design"])
def test_payload_and_capabilities(monkeypatch, tmp_path, definition, model_type):
    # Non-Base modes must not even resolve a stale stored reference, let alone decode it.
    project = make_project(tmp_path, model_type)
    support = AudioCppModelSupport(definition)
    base = model_type == "base"
    assert support.can_voice_clone(project) is base
    assert support.can_voice_presets(project) is (model_type == "custom_voice")
    assert support.requires_voice_preset(project) is (model_type == "custom_voice")
    assert support.can_instructions(project) is (not base)
    assert support.expects_voice_instructions(project) is (model_type == "voice_design")
    if base:
        sf.write(tmp_path / "sample.flac", np.full(100, .2), 24000, format="FLAC")
    else:
        monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *_: pytest.fail("sample queried"))
        monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *_: pytest.fail("sample resolved"))
        monkeypatch.setattr(AudioCppUtil, "make_voice_ref", lambda *_: pytest.fail("sample read"))
    sent = capture_requests(monkeypatch)
    exact_id = "/opaque/checkpoints/selected-Q4.gguf"
    result = AudioCppBackendAdapter(definition, support, exact_id).generate_using_project(project, ["hello"])
    assert not isinstance(result, str), result
    payload = sent[0]
    expected = {"model": exact_id, "input": "hello", "response_format": "wav", "seed": 7,
                "language": "english", "temperature": .9, "top_k": 50, "top_p": 1., "repetition_penalty": 1.05,
                "options": {"text_chunk_size": 100000,
                            "subtalker_temperature": .9, "subtalker_top_k": 50, "subtalker_top_p": 1.}}
    if base:
        assert payload.pop("voice_ref")["data"].startswith("data:audio/wav;base64,")
        expected["options"]["x_vector_only_mode"] = False
        assert payload["options"]["x_vector_only_mode"] is False
        expected["reference_text"] = "spoken words"
    else:
        expected["options"]["instruction"] = "Speak warmly."
        if model_type == "custom_voice":
            expected["options"]["voice_id"] = "vivian"
    # Exact equality excludes selectors and max_tokens; the main talker's sampling
    # controls are mirrored to the subtalker so they govern every codebook.
    assert payload == expected


def test_sampling_controls_mirror_to_subtalker(monkeypatch, tmp_path, definition):
    # Non-default values prove the subtalker follows the user, not the server's own defaults.
    project = make_project(tmp_path, "voice_design")
    project.set_model_setting(MODEL_ID, "temperature", .3)
    project.set_model_setting(MODEL_ID, "top_k", 1)
    project.set_model_setting(MODEL_ID, "top_p", .8)
    sent = capture_requests(monkeypatch)
    support = AudioCppModelSupport(definition)
    result = AudioCppBackendAdapter(definition, support, "qwen3.gguf").generate_using_project(project, ["hello"])
    assert not isinstance(result, str), result
    payload = sent[0]
    assert (payload["temperature"], payload["top_k"], payload["top_p"]) == (.3, 1, .8)
    options = payload["options"]
    assert (options["subtalker_temperature"], options["subtalker_top_k"], options["subtalker_top_p"]) == (.3, 1, .8)


def test_retired_embedding_setting_loads_as_base_icl(monkeypatch, tmp_path, definition):
    # Old projects must not revive embedding-only cloning or re-save its retired selector.
    payload = ProjectSerializationUtil.to_project_json_dict(make_project(tmp_path))
    payload["model_settings"]["models"][MODEL_ID]["parameters"] = {"clone_mode": "x_vector_only"}
    project = Project.model_validate(payload)
    assert "clone_mode" not in AudioCppModelSupport(definition).get_values(project)
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "clone_mode" not in saved["model_settings"]["models"][MODEL_ID].get("parameters", {})
    assert Tts.requires_reference_transcript(project)
    sf.write(tmp_path / "sample.flac", np.full(100, .2), 24000, format="FLAC")
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "bound")
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    assert sent[0]["options"]["x_vector_only_mode"] is False
    assert sent[0]["reference_text"] == "spoken words"
    assert "clone_mode" not in sent[0] and "clone_mode" not in sent[0]["options"]
    project.voice_references[0] = replace(project.voice_references[0], transcript="")
    assert is_missing_required_transcript(project)
    assert adapter.generate_using_project(project, ["hello"]) == "Voice clone transcript required when a voice clone sample is supplied"
    assert len(sent) == 1


def test_base_requires_sample(monkeypatch, tmp_path, definition):
    project = make_project(tmp_path)
    project.set_model_setting(MODEL_ID, "file_name", [])
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "bound")
    assert adapter.generate_using_project(project, ["hello"]) == "A voice clone sample is required"
    assert sent == []


def test_custom_voice_requires_preset(monkeypatch, tmp_path, definition):
    project = make_project(tmp_path, "custom_voice")
    project.voice_references[0].presets.clear()
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "bound")
    assert adapter.generate_using_project(project, ["hello"]) == "A speaker preset is required"
    assert not sent


def test_selected_voice_instructions_and_preset(monkeypatch, tmp_path, definition):
    # Stored instructions/presets are per selected voice, not always voice zero.
    project = make_project(tmp_path, "custom_voice")
    project.voice_references.append(type(project.voice_references[0])(
        file_name="missing.flac", presets={MODEL_ID: "opaque-speaker"}, instructions={MODEL_ID: "Second voice."}))
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "second-bound-id")
    assert not isinstance(adapter.generate_using_project(project, ["hello"], voice_selection_index=1), str)
    assert sent[0]["options"]["voice_id"] == "opaque-speaker"
    assert sent[0]["options"]["instruction"] == "Second voice."


def test_retired_max_tokens_setting_is_dropped(monkeypatch, tmp_path, definition):
    # Projects saved with the former max_tokens override load cleanly, never re-save it,
    # and leave the server on the checkpoint's own generation limit.
    payload = ProjectSerializationUtil.to_project_json_dict(make_project(tmp_path, "voice_design"))
    payload["model_settings"]["models"][MODEL_ID].setdefault("parameters", {})["max_tokens"] = 123
    project = Project.model_validate(payload)
    assert "max_tokens" not in AudioCppModelSupport(definition).get_values(project)
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert "max_tokens" not in saved["model_settings"]["models"][MODEL_ID].get("parameters", {})
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "bound")
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    assert "max_tokens" not in sent[0] and "max_tokens" not in sent[0]["options"]


@pytest.mark.parametrize("model_type,uses_transcript", [
    ("base", True), ("custom_voice", False), ("voice_design", False),
])
def test_menu_preflight_server_and_adapter_agree(monkeypatch, tmp_path, definition, model_type, uses_transcript):
    # Base always uses ICL: an empty transcript blocks every generation entry point.
    project = make_project(tmp_path, model_type, transcript="")
    sf.write(tmp_path / "sample.flac", np.full(100, .2), 24000, format="FLAC")
    state = SimpleNamespace(project=project, prefs=SimpleNamespace(stt_variant=None, stt_config={}))
    labels = [get_string_from(state, i.label).lower() for i in ModelAudioCppMenu.make_items(state, definition)]
    assert not any("clone mode" in label for label in labels)
    assert not any("instructions" in label for label in labels)
    assert Tts.requires_reference_transcript(project) is uses_transcript
    assert is_missing_required_transcript(project) is uses_transcript
    assert VoiceMenuShared.get_voice_problems(state) == (["Voice file sample.flac has no transcript"] if uses_transcript else [])
    transcribed = []
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", lambda *args: transcribed.append(args) or ("", "stt broke"))
    monkeypatch.setattr(voice_menu_shared, "printt", lambda *_: None)
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", lambda *_: None)
    assert VoiceMenuShared.validate_voices(state) is (not uses_transcript)
    assert bool(transcribed) is uses_transcript
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "bound")
    result = adapter.generate_using_project(project, ["hello"])
    if uses_transcript:
        assert result == "Voice clone transcript required when a voice clone sample is supplied" and not sent
    else:
        assert not isinstance(result, str), result
        assert "reference_text" not in sent[0]


def test_base_selected_sample_and_transcript(monkeypatch, tmp_path, definition):
    # A selected nonzero voice must supply both the audio and its matching transcript.
    project = make_project(tmp_path)
    project.voice_references.append(type(project.voice_references[0])(
        file_name="second.flac", transcript="second words"))
    resolved = []
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path",
                        lambda project, name: resolved.append(name) or name)
    monkeypatch.setattr(AudioCppUtil, "make_voice_ref", lambda path: "encoded:" + path)
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "bound-second")
    assert not isinstance(adapter.generate_using_project(project, ["hello"], voice_selection_index=1), str)
    assert resolved == ["second.flac"]
    assert sent[0]["voice_ref"]["data"] == "encoded:second.flac"
    assert sent[0]["reference_text"] == "second words"


def test_voice_design_missing_instructions_is_a_hint_not_a_blocker(monkeypatch, tmp_path, definition):
    project = make_project(tmp_path, "voice_design")
    project.voice_references[0].instructions.clear()
    support = AudioCppModelSupport(definition)
    assert support.expects_voice_instructions(project)
    assert not [issue for issue in support.get_blocking_issues(project) if issue.short != "audio.cpp server"]
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, support, "design-bound")
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    assert "instruction" not in sent[0]["options"]


@pytest.mark.parametrize("language,expected", [
    ("EN-us", "english"), ("Chinese", "chinese"), ("zh-CN", "chinese"),
    ("unsupported", "auto"), ("", "auto"),
])
def test_language_payload_and_warning_agree(monkeypatch, tmp_path, definition, language, expected):
    # Older servers reject ISO hints; warnings must describe the actual full-name/auto value sent.
    project = make_project(tmp_path, "voice_design")
    project.language_code = language
    support = AudioCppModelSupport(definition)
    sent = capture_requests(monkeypatch)
    adapter = AudioCppBackendAdapter(definition, support, "language-bound")
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    assert sent[0]["language"] == expected
    assert [warning for warning in support.get_warning_issues(project)
            if warning.startswith("Language value passed to model:")] == [make_language_value_note(expected, language)]


def test_discovery_keeps_all_four_exact_choices():
    # Names/path fragments are opaque and must not infer checkpoint type or pick the first match.
    ids = ["operator-base", "/models/opaque.gguf", "custom-configured", "design-Q8"]
    entries = [{"id": model_id, "family": "qwen3_tts", "task": task, "mode": "offline"}
               for model_id, task in zip(ids, ("tts", "clon", "tts", "vdes"))]
    handle = TtsModelType.require_by_id(MODEL_ID)
    assert detect_audio_cpp_models(entries) == [(handle, model_id) for model_id in ids]
    for change in ({"task": "asr"}, {"mode": "streaming"}, {"family": "qwen3"}):
        assert detect_audio_cpp_models([{**entries[0], **change}]) == []


def test_private_settings_and_voice_data_roundtrip(tmp_path, definition):
    # audio.cpp defaults/overrides must not mutate local or SGL Qwen settings.
    project = make_project(tmp_path, "voice_design")
    project.set_model_setting(MODEL_ID, "temperature", 1.2)
    project.set_model_setting("qwen3tts_local", "temperature", .6)
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    parameters = payload["model_settings"]["models"][MODEL_ID]["parameters"]
    assert parameters["temperature"] == 1.2
    assert "clone_mode" not in parameters
    restored = Project.model_validate(payload)
    assert restored.voice_references == project.voice_references
    assert AudioCppModelSupport(definition).get_values(restored) == AudioCppModelSupport(definition).get_values(project)
    assert restored.get_model_setting("qwen3tts_local", "temperature") == .6
    assert restored.get_model_setting("qwen3tts_sglomni", "temperature") == -1
    assert restored.get_model_setting(MODEL_ID, "seed") == 7
