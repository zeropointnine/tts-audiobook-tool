"""Catalog edit-prompt hints do not alter menu labels or persisted settings."""
from types import SimpleNamespace

import pytest

from catalog_toml_support import read_catalog, write_catalog
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
from tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu import VoiceConfiguredSglOmniMenu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions


def test_breeze_top_k_has_prompt_only_disabled_hint(monkeypatch):
    definition = load_audio_cpp_definitions().models["breeze_tts_2_audiocpp"]
    parameter = definition.parameters["top_k"]
    control = next(control for control in definition.menu if control.parameter == "top_k")
    assert parameter.input_prompt_suffix == "(0=disabled)"
    assert control.label == "Top-K"
    state = SimpleNamespace(project=Project())
    item = VoiceAudioCppMenu.make_parameter_item(state, parameter, control.label)
    prompts = []
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_audio_cpp_menu.printt", prompts.append)
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_audio_cpp_menu.ask.ask_input", lambda **_: "")
    assert strip_ansi_codes(get_string_from(state, item.label)) == "Top-K (currently: 50 default)"
    item.handler(state, item)
    assert prompts == [f"Enter Top-K (valid range: 0-100; default: 50) {COL_DIM}(0=disabled){COL_DEFAULT}:"]
    saved = ProjectSerializationUtil.to_project_json_dict(state.project)
    assert "input_prompt_suffix" not in saved["model_settings"]["models"][parameter.model_id]["parameters"]


def _load_parameter(tmp_path, backend, suffix):
    data = read_catalog(CATALOG_PATH)
    model_id = "breeze_tts_2_audiocpp" if backend == "audio_cpp" else "zonos2_sglomni"
    entry = next(entry for entry in data["models"] if entry["id"] == model_id)
    raw = entry[backend]["parameters"]["top_k"]
    if suffix is None:
        raw.pop("input_prompt_suffix", None)
    else:
        raw["input_prompt_suffix"] = suffix
    path = write_catalog(tmp_path / "catalog.toml", data)
    loader = load_audio_cpp_definitions if backend == "audio_cpp" else load_definitions
    return loader(path).models[model_id].parameters["top_k"]


@pytest.mark.parametrize("backend", ["audio_cpp", "sgl_omni"])
@pytest.mark.parametrize("suffix", [None, "", "(sampling hint)"])
def test_optional_suffix_is_loaded_and_inserted_only_after_range(tmp_path, monkeypatch, backend, suffix):
    parameter = _load_parameter(tmp_path, backend, suffix)
    assert parameter.input_prompt_suffix == (suffix or "")
    state = SimpleNamespace(project=Project())
    menu = VoiceAudioCppMenu if backend == "audio_cpp" else VoiceConfiguredSglOmniMenu
    module = "voice_audio_cpp_menu" if backend == "audio_cpp" else "voice_configured_sgl_omni_menu"
    prompts = []
    monkeypatch.setattr(f"tts_audiobook_tool.menus.voice.{module}.printt", prompts.append)
    monkeypatch.setattr(f"tts_audiobook_tool.menus.voice.{module}.ask.ask_input", lambda **_: "")
    item = menu.make_parameter_item(state, parameter, "Top k")
    assert "sampling hint" not in get_string_from(state, item.label)
    item.handler(state, item)
    hint = f" {COL_DIM}{suffix}{COL_DEFAULT}" if suffix else ""
    if backend == "audio_cpp":
        expected = f"Enter Top k (valid range: 0-100; default: 50){hint}:"
    else:
        expected = f"Enter Top k (valid range: 1-200; -1 resets to default 100){hint}:"
    assert prompts == [expected]


@pytest.mark.parametrize("backend", ["audio_cpp", "sgl_omni"])
@pytest.mark.parametrize("invalid", [123, False, [], {}])
def test_suffix_must_be_a_string(tmp_path, backend, invalid):
    with pytest.raises(ValueError, match=r"top_k\.input_prompt_suffix.*expected a string"):
        _load_parameter(tmp_path, backend, invalid)
