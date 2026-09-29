"""Higgs configured path; legacy/project formats remain unchanged."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from catalog_toml_support import read_catalog, write_catalog
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu import VoiceConfiguredSglOmniMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings, ConfiguredModelSupport, SglOmniBackendAdapter
from tts_audiobook_tool.tts_models.sgl_omni_definition import DEFINITION_PATH, load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from project_settings_test_support import set_setting


@pytest.fixture
def configured(monkeypatch):
    old = (Tts._backend_mode, getattr(Tts, "_type", None),
           Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsBackendKind.SGL_OMNI))
    try:
        Tts.init_local_model_type()
        Tts._type = TtsModelType.HIGGS_V3_SERVER
        yield Tts.get_configured_definition()
    finally:
        TtsModelType.reset_catalog()
        REGISTRY.reset_to_builtins()
        (Tts._backend_mode, previous_type,
         Tts._config_fingerprint, Tts._configured_definitions, Tts._configured_runtime, Tts._catalog_initialized) = old
        if previous_type is None:
            if hasattr(Tts, "_type"):
                delattr(Tts, "_type")
        else:
            Tts._type = previous_type


def test_loaded_definition_and_registry_overlay(configured):
    definition = configured
    assert definition.spec.id == "server_higgs_v3"
    assert definition.spec.default_output_sample_rate == 24_000
    assert definition.spec.substitutions == [("—", ", "), ("─", ", ")]
    assert Tts.get_info() is definition.spec
    assert TtsModelType.HIGGS_V3_SERVER.value is definition.spec
    assert TtsModelType.get_by_id(definition.spec.id) is TtsModelType.HIGGS_V3_SERVER
    assert TtsModelType.all().count(TtsModelType.HIGGS_V3_SERVER) == 1
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id("BOSON/HIGGS") is TtsModelType.HIGGS_V3_SERVER
    assert TtsModelType("server_higgs_v3") == TtsModelType.HIGGS_V3_SERVER
    assert hash(TtsModelType("server_higgs_v3")) == hash(TtsModelType.HIGGS_V3_SERVER)
    assert isinstance(Tts.get_model_support(), ConfiguredModelSupport)
    assert isinstance(Tts.get_instance(), SglOmniBackendAdapter)
    assert Tts.get_instance() is Tts.get_instance_if_exists()
    assert Tts.instance_exists()


def test_loader_rejects_invalid_values_without_installing_overlay(tmp_path):
    source = read_catalog(DEFINITION_PATH)
    old_spec = TtsModelType.HIGGS_V3_SERVER.value
    for path, value, error in [
        (("parameters", "top_k", "default"), 101, "top_k.default"),
        (("parameters", "top_p", "min"), True, "top_p.min"),
        (("parameters", "temperature", "request_key"), "input", "temperature.request_key"),
        (("menu", 2, "parameter"), "no_such_param", r"menu\[2\]\.parameter"),
        (("spec", "voice_target_attr"), "unpersisted_voice", "voice_target_attr"),
        (("spec", "voice_transcript_attr"), "qwen3_voice_file_name", "voice_transcript_attr"),
        (("spec", "batch_size_attr"), "mira_batch_size", "batch_size_attr"),
        (("behavior", "orchestration"), "batch_size", "behavior.orchestration"),
        (("parameters", "temperature", "project_attr"), "higgs_v3_temperature", "project_attr"),
        (("parameters", "temperature", "default_sentinel"), -2, "temperature.default_sentinel"),
        (("parameters", "top_p", "default_sentinel"), -2, "top_p.default_sentinel"),
        (("parameters", "top_k", "default_sentinel"), -2, "top_k.default_sentinel"),
        (("behavior", "seed"), "always", "behavior"),
    ]:
        data = copy.deepcopy(source)
        target = next(entry for entry in data["models"] if entry["id"] == "server_higgs_v3")
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        file = write_catalog(tmp_path / "bad.toml", data)
        with pytest.raises(ValueError, match=error):
            load_definitions(file)
        assert TtsModelType.HIGGS_V3_SERVER.value is old_spec


def test_definition_v1_requires_explicit_upgrade(tmp_path):
    source = read_catalog(DEFINITION_PATH)
    source["schema_version"] = 1
    path = write_catalog(tmp_path / "old.toml", source)
    with pytest.raises(ValueError, match="schema_version.*version 3"):
        load_definitions(path)


def test_local_mode_has_no_server_runtime_definitions(monkeypatch):
    original = (Tts._backend_mode, getattr(Tts, "_type", None),
                Tts._configured_definitions, Tts._configured_runtime,
                Tts._catalog_initialized, Tts._config_fingerprint)
    try:
        monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsBackendKind.LOCAL))
        Tts.init_local_model_type()
        assert not Tts._configured_definitions
        assert Tts._configured_runtime is None
        assert not Tts._config_fingerprint
        assert Tts.get_configured_definition(TtsModelType.HIGGS_V3_SERVER) is None
    finally:
        TtsModelType.reset_catalog()
        (Tts._backend_mode, previous_type,
         Tts._configured_definitions, Tts._configured_runtime,
         Tts._catalog_initialized, Tts._config_fingerprint) = original
        if previous_type is None:
            if hasattr(Tts, "_type"):
                delattr(Tts, "_type")
        else:
            Tts._type = previous_type


def test_invalid_config_fails_at_startup(monkeypatch):
    previous = (Tts._backend_mode, Tts._configured_definitions, Tts._config_fingerprint, Tts._catalog_initialized)
    monkeypatch.setattr(Tts, "_probe_backend_mode", staticmethod(lambda: TtsBackendKind.SGL_OMNI))
    def fail():
        raise ValueError("model server_higgs_v3.parameters.top_k: invalid value")
    monkeypatch.setattr("tts_audiobook_tool.tts_models.sgl_omni_definition.load_definitions", fail)
    try:
        with pytest.raises(ValueError, match="server_higgs_v3.parameters.top_k"):
            Tts.init_local_model_type()
        assert TtsModelType.HIGGS_V3_SERVER.value is TtsModelType._builtin_specs["server_higgs_v3"]
    finally:
        TtsModelType.reset_catalog()
        (Tts._backend_mode, Tts._configured_definitions,
         Tts._config_fingerprint, Tts._catalog_initialized) = previous


def test_settings_and_project_format_use_model_settings(configured, tmp_path, monkeypatch):
    project = Project.model_validate({"dir_path": str(tmp_path), "higgs_v3_seed": 77})
    params = configured.parameters
    assert [ConfiguredSettings.get(project, p) for p in params.values()] == [1.0, 1.0, 100]
    # A value that merely follows the declared default is absent, never the
    # legacy -1 sentinel.
    stored = project.model_settings.models["server_higgs_v3"]["parameters"]
    assert "temperature" not in stored and "top_p" not in stored and "top_k" not in stored
    monkeypatch.setattr(Project, "save", lambda self: "")
    assert ConfiguredSettings.set(project, params["temperature"], 0.7) == ""
    assert ConfiguredSettings.set(project, params["top_k"], 42) == ""
    assert ConfiguredSettings.set(project, params["top_k"], None) == ""
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    parameters = payload["model_settings"]["models"]["server_higgs_v3"]["parameters"]
    assert parameters["temperature"] == 0.7
    assert "top_k" not in parameters
    assert parameters["seed"] == 77
    reloaded = Project.model_validate(payload)
    assert ConfiguredSettings.get(reloaded, params["temperature"]) == 0.7
    assert reloaded.get_model_setting("server_higgs_v3", "seed") == 77


@pytest.mark.parametrize("name, entry, expected", [
    ("temperature", "0.7", 0.7),
    ("top_p", "0.8", 0.8),
    ("top_k", "42", 42),
])
def test_numeric_menu_edit_and_reset_use_existing_project_keys(configured, monkeypatch, tmp_path, name, entry, expected):
    project = Project.model_validate({"dir_path": str(tmp_path), "higgs_v3_seed": 77})
    state = SimpleNamespace(project=project)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *_: [])
    item = VoiceConfiguredSglOmniMenu.make_items(state, configured)[list(configured.parameters).index(name)]
    saved = []
    monkeypatch.setattr(Project, "save", lambda self: saved.append(ProjectSerializationUtil.to_project_json_dict(self)) or "")
    responses = iter((entry, "-1"))
    prefilled = []
    def ask_input(*, prefill):
        prefilled.append(prefill)
        return next(responses)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_input", ask_input)
    errors = []
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_error", errors.append)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.print_feedback", lambda *_: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.printt", lambda *_: None)

    item.handler(state, item)
    stored = project.model_settings.models["server_higgs_v3"]["parameters"]
    assert stored[name] == expected
    saved_parameters = saved[0]["model_settings"]["models"]["server_higgs_v3"]["parameters"]
    assert saved_parameters[name] == expected
    assert saved_parameters["seed"] == 77
    assert prefilled[0] == str(configured.parameters[name].default)
    assert str(expected) in item.label(state)

    item.handler(state, item)
    assert name not in project.model_settings.models["server_higgs_v3"]["parameters"]
    assert name not in saved[1]["model_settings"]["models"]["server_higgs_v3"]["parameters"]
    assert len(saved) == 2
    assert prefilled[1] == str(expected)
    assert "default" in item.label(state)
    assert not errors


@pytest.mark.parametrize("name, entry", [
    ("temperature", "0"),
    ("top_p", "1.1"),
    ("top_k", "1.5"),
    ("top_k", "101"),
    ("temperature", "nan"),
    ("temperature", "not a number"),
])
def test_numeric_menu_rejects_invalid_edits_without_saving(configured, monkeypatch, name, entry):
    state = SimpleNamespace(project=Project())
    item = VoiceConfiguredSglOmniMenu.make_parameter_item(state, configured.parameters[name], name)
    saves = []
    monkeypatch.setattr(Project, "save", lambda self: saves.append(True) or "")
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_input", lambda **_: entry)
    errors = []
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_error", errors.append)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.printt", lambda *_: None)

    item.handler(state, item)
    stored = state.project.model_settings.models.get("server_higgs_v3", {}).get("parameters", {})
    assert name not in stored
    assert not saves
    assert len(errors) == 1


def test_numeric_menu_can_reset_invalid_stored_value(configured, monkeypatch):
    project = Project()
    set_setting(project, "higgs_v3_top_k", 101)
    state = SimpleNamespace(project=project)
    item = VoiceConfiguredSglOmniMenu.make_parameter_item(state, configured.parameters["top_k"], "Top_K")
    assert "invalid" in item.label(state)
    monkeypatch.setattr(Project, "save", lambda self: "")
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_input", lambda **_: "-1")
    errors = []
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.ask.ask_error", errors.append)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.print_feedback", lambda *_: None)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu.printt", lambda *_: None)

    item.handler(state, item)
    assert ConfiguredSettings.get(project, configured.parameters["top_k"]) == 100
    assert len(errors) == 1


def test_adapter_payload_reference_and_callbacks(configured, monkeypatch, tmp_path):
    from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
    from tts_audiobook_tool.sound.sound_util import SoundUtil
    first = tmp_path / "first.flac"
    second = tmp_path / "second.flac"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    monkeypatch.setattr(SglOmniUtil, "get_base_url", lambda: "http://example.test")
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", lambda path: "data:" + path)
    calls = []
    sound = Sound(np.array([0.1], dtype=np.float32), 24_000)
    def concurrent(base, payloads, **kwargs):
        calls.append((base, payloads, kwargs))
        return [sound for _ in payloads]
    monkeypatch.setattr(SglOmniUtil, "generate_concurrent", concurrent)
    project = Project.model_validate({"dir_path": str(tmp_path), "higgs_v3_voice_file_name": ["first.flac", "second.flac"],
        "higgs_v3_voice_transcript": ["one", "two"], "higgs_v3_seed": 42,
        "higgs_v3_temperature": 0.7, "higgs_v3_top_p": 0.8, "higgs_v3_top_k": 12})
    instance = Tts.get_instance()
    assert instance.generate_using_project(project, ["ready A", "ready B"], voice_selection_index=1, force_random_seed=True) == [sound, sound]
    assert calls[0] == ("http://example.test", [{"max_tokens": 1536, "temperature": 0.7, "top_p": 0.8, "top_k": 12,
        "input": prompt, "stream": False, "references": [{"audio_path": "data:" + str(second), "text": "two"}]}
        for prompt in ("ready A", "ready B")], {"print_request": False, "fallback_sample_rate": 24000})
    assert not any("seed" in payload for payload in calls[0][1])
    assert instance.generate_using_project(project, ["a", "b"], on_stream_end=lambda: None) == "Streaming generation supports exactly one prompt"
    streamed = []
    def streaming(base, payload, **kwargs):
        streamed.append(payload)
        kwargs["on_stream_chunk"](sound.data)
        kwargs["on_stream_end"]()
        return sound
    monkeypatch.setattr(SglOmniUtil, "generate_streaming", streaming)
    callbacks = []
    assert instance.generate_using_project(project, ["ready"], on_stream_chunk=lambda data: callbacks.append("chunk"),
        on_stream_end=lambda: callbacks.append("end")) == [sound]
    assert callbacks == ["chunk", "end"]
    assert streamed[0]["stream"] is True
    Tts.clear_tts_model()
    assert not Tts.instance_exists()


def test_voice_menu_routes_to_definition_not_legacy_menu(configured, monkeypatch):
    calls = []
    monkeypatch.setattr(VoiceConfiguredSglOmniMenu, "menu", lambda state, definition: calls.append(definition))
    VoiceMenuShared.menu(SimpleNamespace(project=Project()))
    assert calls == [configured]


def test_decoding_uses_metadata_before_configured_fallback(monkeypatch):
    from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
    from tts_audiobook_tool.app_support import sgl_omni_util
    monkeypatch.setattr(sgl_omni_util.soundfile, "read", lambda *args, **kwargs: (np.array([0.2], dtype=np.float32), 11025))
    assert SglOmniUtil.sound_from_encoded_audio(b"audio", fallback_sample_rate=24000).sr == 11025
    monkeypatch.setattr(sgl_omni_util.soundfile, "read", lambda *args, **kwargs: (np.array([0.2], dtype=np.float32), 0))
    assert SglOmniUtil.sound_from_encoded_audio(b"audio", fallback_sample_rate=24000).sr == 24000


def test_configured_readiness_and_menu(configured, monkeypatch, tmp_path):
    from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
    monkeypatch.setattr(SglOmniUtil, "check_readiness", lambda _: None)
    project = Project.model_validate({"dir_path": str(tmp_path), "higgs_v3_voice_file_name": ["missing.flac"],
                                      "higgs_v3_voice_transcript": [""]})
    issues = Tts.get_model_support().get_blocking_issues(project, None)
    assert {issue.short for issue in issues} >= {"voice sample", "voice clone transcript"}
    set_setting(project, "higgs_v3_top_k", 101)
    assert any(issue.short == "top_k" for issue in Tts.get_model_support().get_blocking_issues(project, None))
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *args: [MenuItem("voice", lambda *_: None)])
    items = VoiceConfiguredSglOmniMenu.make_items(SimpleNamespace(project=Project()), configured)
    assert len(items) == 4
    assert "Temperature" in items[1].label(SimpleNamespace(project=Project()))
    assert "Top-P" in items[2].label(SimpleNamespace(project=Project()))
    assert "Top-K" in items[3].label(SimpleNamespace(project=Project()))
    assert items[1].superlabel
