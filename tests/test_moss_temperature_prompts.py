"""Shared temperature prompt contracts for both MOSS variants on all backends."""
from types import SimpleNamespace

import pytest

from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
from tts_audiobook_tool.menus.model.model_configured_sgl_omni_menu import ModelConfiguredSglOmniMenu
from tts_audiobook_tool.menus.model.model_moss_shared import ModelMossShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions


CASES = (
    ("local", "delay", "moss_local", 1.7, "MossTTSDelay temperature"),
    ("local", "local", "moss_local", 1.2, "MossTTSLocal temperature"),
    ("local", "local_v15", "moss_local", 1.7, "MossTTSLocal v1.5 temperature"),
    ("sgl_omni", "delay", "moss_delay_sglomni", 1.7, "Temperature"),
    ("sgl_omni", "local", "moss_local_sglomni", 1.0, "Temperature"),
    ("audio_cpp", "delay", "moss_delay_audiocpp", 1.5, "Temperature"),
    ("audio_cpp", "local", "moss_local_audiocpp", 1.7, "Temperature"),
)


def make_temperature_item(state, backend, architecture):
    if backend == "local":
        config = {
            "delay": MossConfigs.DELAY,
            "local": MossConfigs.LOCAL,
            "local_v15": MossConfigs.LOCAL_V15,
        }[architecture]
        state.project.set_model_setting("moss_local", "target", config.value.repo_id)
        return ModelMossShared.make_temperature_item(state, config)
    model_id = state.project.tts_model_type
    if backend == "sgl_omni":
        definition = load_definitions().models[model_id]
        return ModelConfiguredSglOmniMenu.make_items(state, definition)[0]
    definition = load_audio_cpp_definitions().models[model_id]
    return ModelAudioCppMenu.make_items(state, definition)[0]


@pytest.fixture
def numeric_ui(monkeypatch):
    prompts, errors, saves, shown_hints = [], [], [], []
    monkeypatch.setattr(ask, "printt", prompts.append)
    monkeypatch.setattr(ask, "ask_error", errors.append)
    monkeypatch.setattr(ask, "print_feedback", lambda *_: None)
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))
    monkeypatch.setattr(hints, "show_hint_if_necessary", lambda prefs, hint: shown_hints.append(hint))
    return prompts, errors, saves, shown_hints


@pytest.mark.parametrize("backend,architecture,model_id,default,label", CASES)
@pytest.mark.parametrize("response", ["", "unchanged"])
def test_all_moss_temperatures_have_standard_prompt_and_preserve_saved_defaults(
        monkeypatch, numeric_ui, backend, architecture, model_id, default, label, response):
    prompts, errors, saves, shown_hints = numeric_ui
    state = SimpleNamespace(project=Project(tts_model_type=model_id), prefs=object())
    name = f"{architecture}_temperature"
    prefills = []

    def answer(*, prefill):
        prefills.append(prefill)
        return prefill if response == "unchanged" else response

    monkeypatch.setattr(ask, "ask_input", answer)
    item = make_temperature_item(state, backend, architecture)
    item.handler(state, item)
    assert [text_util.strip_ansi_codes(prompt) for prompt in prompts] == [
        f"Enter {label}: (valid range: 0.8-3.0; default: {default})"]
    assert prefills == [str(default)]
    assert state.project.get_model_setting(model_id, name) == -1
    assert errors == [] and saves == []
    assert shown_hints == [HINT_MOSS_TEMPERATURE]
    assert text_util.strip_ansi_codes(get_string_from(state, item.label)) == (
        f"{label} (currently: {default:.2f} default)")


@pytest.mark.parametrize("backend,architecture,model_id,default,label", CASES)
@pytest.mark.parametrize("response,expected,error", [
    ("-1", 2.3, "Out of range"),
    ("invalid", 2.3, "Bad value"),
    ("nan", 2.3, "Out of range"),
    ("2.3", 2.3, ""),
    ("0.8", 0.8, ""),
    ("3.0", 3.0, ""),
    ("default", None, ""),
])
def test_all_moss_temperatures_use_stock_validation_and_no_minus_one_reset(
        monkeypatch, numeric_ui, backend, architecture, model_id, default, label, response, expected, error):
    prompts, errors, saves, _ = numeric_ui
    project = Project(tts_model_type=model_id)
    name = f"{architecture}_temperature"
    project.set_model_setting(model_id, name, 2.3)
    state = SimpleNamespace(project=project, prefs=object())
    item = make_temperature_item(state, backend, architecture)
    prefills = []

    def answer(*, prefill):
        prefills.append(prefill)
        return str(default) if response == "default" else response

    monkeypatch.setattr(ask, "ask_input", answer)
    item.handler(state, item)
    expected = default if expected is None else expected
    assert prefills == ["2.3"]
    assert errors == ([error] if error else [])
    assert saves == ([] if expected == 2.3 else [project])
    assert project.get_model_setting(model_id, name) == expected
    assert "resets" not in prompts[0] and "-1" not in prompts[0]
