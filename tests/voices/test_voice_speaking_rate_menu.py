"""Per-voice rate override menu, transactional edits, and capability isolation."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice import voice_menu_shared as shared
from tts_audiobook_tool.menus.voice import voice_speaking_rate_items as rate_menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts

MODEL_ID = "zonos3_mini_sglomni"


@pytest.fixture(autouse=True)
def isolate_menu(monkeypatch):
    monkeypatch.setattr(shared.AudioMetaUtil, "get_audio_duration", lambda _: None)
    monkeypatch.setattr(rate_menu, "print_feedback", lambda *_: None)
    monkeypatch.setattr(Project, "save", lambda _: pytest.fail("Unexpected save"))
    monkeypatch.setattr(Tts, "get_instance", lambda: pytest.fail("Menu must not load a model"))


def make_state(model_id=MODEL_ID):
    project = Project(tts_model_type=model_id, voice_references=[
        Voice("sample.flac", "Keep transcript", instructions={"other_model": "Keep instructions"},
              speaking_rate_buckets={MODEL_ID: 7, "other_model": 12}),
        Voice(),
    ])
    return cast(State, SimpleNamespace(project=project))


def items(state, menu):
    value = menu["items"]
    return value(state) if callable(value) else value


def label(state, item):
    return strip_ansi_codes(get_string_from(state, item.label))


def select(state, menu, prefix):
    item = next(item for item in items(state, menu) if label(state, item).startswith(prefix))
    item.handler(state, item)


def open_submenu(monkeypatch, state, index=0):
    captured = {}
    monkeypatch.setattr(shared.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    shared.VoiceMenuShared.menu(state)
    outer = dict(captured)
    select(state, outer, f"Voice {index + 1}:")
    return outer, dict(captured)


@pytest.mark.parametrize("model_id,shown", [(MODEL_ID, True), ("zonos2_sglomni", False), ("qwen3tts_audiocpp", False)])
def test_rate_control_is_per_voice_and_only_for_supported_model(monkeypatch, model_id, shown):
    # Enter via real catalog dispatch; retained data alone cannot enable a control.
    state = make_state(model_id)
    original = list(state.project.voice_references)
    outer, submenu = open_submenu(monkeypatch, state)
    assert not any(label(state, item).startswith(rate_menu.LABEL_SPEAKING_RATE) for item in items(state, outer))
    controls = [item for item in items(state, submenu) if label(state, item).startswith(rate_menu.LABEL_SPEAKING_RATE)]
    assert bool(controls) is shown
    if shown:
        assert "Bucket 7" in label(state, controls[0])
        _, second = open_submenu(monkeypatch, state, index=1)
        assert any("Disabled" in label(state, item) for item in items(state, second)
                   if label(state, item).startswith(rate_menu.LABEL_SPEAKING_RATE))
    assert state.project.voice_references == original


@pytest.mark.parametrize("answer,expected", [("1", 1), ("30", 30), ("0", None)])
def test_rate_edit_changes_only_selected_voice_and_model(monkeypatch, answer, expected):
    # Numeric endpoints and disabling preserve the sample and other model-specific data.
    state = make_state()
    before = state.project.voice_references[0]
    saves, prompts = [], []
    monkeypatch.setattr(Project, "save", lambda project: saves.append(project) or "")
    monkeypatch.setattr(rate_menu.ask, "ask_input", lambda **kwargs: prompts.append(kwargs) or answer)
    _, submenu = open_submenu(monkeypatch, state)
    select(state, submenu, rate_menu.LABEL_SPEAKING_RATE)
    after = state.project.voice_references[0]
    assert prompts == [{"prefill": "7"}]
    assert after.speaking_rate_buckets.get(MODEL_ID) == expected
    assert after.speaking_rate_buckets["other_model"] == 12
    assert after.file_name == before.file_name and after.transcript == before.transcript
    assert after.instructions == before.instructions
    assert state.project.voice_references[1] == Voice()
    assert len(saves) == 1
    value = "Disabled" if expected is None else f"Bucket {expected}"
    assert any(value in label(state, item) for item in items(state, submenu)
               if label(state, item).startswith(rate_menu.LABEL_SPEAKING_RATE))


@pytest.mark.parametrize("answer,error", [("", False), ("7", False), ("-1", True), ("31", True), ("7.5", True), ("bad", True)])
def test_cancel_unchanged_and_invalid_rate_inputs_do_not_save(monkeypatch, answer, error):
    # Buckets require whole integers; invalid input must not modify the voice.
    state = make_state()
    original = state.project.voice_references
    errors = []
    monkeypatch.setattr(rate_menu.ask, "ask_input", lambda **_: answer)
    monkeypatch.setattr(rate_menu.ask, "ask_error", errors.append)
    _, submenu = open_submenu(monkeypatch, state)
    select(state, submenu, rate_menu.LABEL_SPEAKING_RATE)
    assert state.project.voice_references is original
    assert bool(errors) is error


def test_enabling_sampleless_voice_and_failed_save(monkeypatch):
    # A new voice starts Disabled; failed saves roll back, then a successful retry applies.
    state = make_state()
    original = state.project.voice_references
    errors, prompts = [], []
    monkeypatch.setattr(Project, "save", lambda _: "Disk full")
    monkeypatch.setattr(rate_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(rate_menu.ask, "ask_input", lambda **kwargs: prompts.append(kwargs) or "4")
    _, submenu = open_submenu(monkeypatch, state, index=1)
    select(state, submenu, rate_menu.LABEL_SPEAKING_RATE)
    assert prompts == [{"prefill": "0"}]
    assert errors == ["Disk full"]
    assert state.project.voice_references is original
    monkeypatch.setattr(Project, "save", lambda _: "")
    select(state, submenu, rate_menu.LABEL_SPEAKING_RATE)
    assert state.project.voice_references[1].speaking_rate_buckets == {MODEL_ID: 4}
    assert state.project.voice_references[0] == original[0]


def test_rate_setter_validates_before_mutating_or_saving():
    # Programmatic callers get the same strict range and index checks as the UI.
    project = make_state().project
    original = project.voice_references
    with pytest.raises(ValueError):
        ProjectVoiceUtil.set_voice_speaking_rate_bucket_at_index_and_save(project, 0, MODEL_ID, 31)
    with pytest.raises(IndexError):
        ProjectVoiceUtil.set_voice_speaking_rate_bucket_at_index_and_save(project, -1, MODEL_ID, 7)
    assert project.voice_references is original
