import pytest

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot, RemoteTtsIssue
from tts_audiobook_tool.constants_hints import (
    HINT_CHATTERBOX_MULTILINGUAL_V3,
    HINT_SGL_OMNI_URL,
)
from tts_audiobook_tool.menus.main_menu import MainMenu, get_heading_tts_text, make_project_label
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from project_settings_test_support import set_setting
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import COL_ERROR


def make_state(model: TtsModelType = TtsModelType.require_by_id("none")) -> State:
    state = object.__new__(State)
    state._prefs = Prefs()
    state._project = Project(dir_path="", tts_model_type=model.id)
    state.has_shown_main_menu = False
    return state


@pytest.mark.parametrize("dir_path", ["", "/example/book"])
@pytest.mark.parametrize("available_count", [0, 1, 2, 3])
def test_project_label_requires_selection_only_with_multiple_types(monkeypatch, dir_path, available_count):
    state = make_state()
    state.project.dir_path = dir_path
    models = [TtsModelType.require_by_id("chatterbox_audiocpp"), TtsModelType.require_by_id("higgs_v3_audiocpp"),
              TtsModelType.require_by_id("echo_tts_audiocpp")]
    monkeypatch.setattr(Tts, "get_available_tts_models", lambda: models[:available_count])

    expected = "Project"
    if available_count >= 2:
        expected += f" {COL_ERROR}(requires: TTS model selection)"
    assert make_project_label(state) == expected


def test_project_label_with_selected_model_does_not_check_availability(monkeypatch):
    state = make_state(TtsModelType.require_by_id("chatterbox_audiocpp"))
    monkeypatch.setattr(Tts, "get_available_tts_models",
                        lambda: pytest.fail("A selected model needs no selection suffix"))

    assert make_project_label(state) == "Project"


def _capture_and_invoke_on_shown(monkeypatch, state) -> dict:
    captured = {}

    def menu(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("tts_audiobook_tool.menus.main_menu.MenuUtil.menu", menu)

    MainMenu.menu(state)

    assert "on_shown" in captured
    captured["on_shown"]()
    return captured


def test_main_menu_on_shown_marks_main_menu_shown(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._backend_mode = TtsRuntimeMode.LOCAL
        state = make_state()
        marks = []
        state.mark_main_menu_shown = lambda: marks.append(True)  # type: ignore[method-assign]
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert marks == [True]


def test_main_menu_on_shown_shows_v3_hint_for_chatterbox_v2(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    state = None
    try:
        Tts._type = TtsModelType.require_by_id("chatterbox_local")
        Tts._backend_mode = TtsRuntimeMode.LOCAL
        state = make_state(TtsModelType.require_by_id("chatterbox_local"))
        set_setting(state.project, "chatterbox_type", ChatterboxType.MULTILINGUAL_V2)
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )

        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == [(state.prefs, HINT_CHATTERBOX_MULTILINGUAL_V3)]


def test_main_menu_on_shown_skips_v3_hint_for_chatterbox_v3(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    try:
        Tts._type = TtsModelType.require_by_id("chatterbox_local")
        Tts._backend_mode = TtsRuntimeMode.LOCAL
        state = make_state(TtsModelType.require_by_id("chatterbox_local"))
        set_setting(state.project, "chatterbox_type", ChatterboxType.MULTILINGUAL_V3)
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )

        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == []


def test_main_menu_on_shown_shows_sgl_omni_url_hint_when_offline_and_unset(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    state = None
    try:
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        Tts._selected_server_model_id = ""
        state = make_state()  # Prefs() leaves sgl_omni_url unset (empty)
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == [(state.prefs, HINT_SGL_OMNI_URL)]


def test_main_menu_on_shown_does_not_show_sgl_omni_url_hint_when_url_set(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    try:
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        Tts._selected_server_model_id = ""
        state = make_state()
        state._prefs = Prefs(remote_tts_url="http://example.test:9009")
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == []


def test_main_menu_on_shown_does_not_show_sgl_omni_url_hint_when_online(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    try:
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        Tts._selected_server_model_id = "bosonai/higgs-audio-v3"
        state = make_state()
        state._prefs = Prefs(remote_tts_url="http://example.test")
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == []


def test_main_menu_on_shown_does_not_show_sgl_omni_url_hint_in_local_mode(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    try:
        Tts._backend_mode = TtsRuntimeMode.LOCAL
        Tts._selected_server_model_id = ""
        state = make_state()
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == []


def test_main_menu_on_shown_does_not_show_sgl_omni_url_hint_after_first_display(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    try:
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        Tts._selected_server_model_id = ""
        state = make_state()
        state.has_shown_main_menu = True
        state.mark_main_menu_shown = lambda: None  # type: ignore[method-assign]
        monkeypatch.setattr(
            hints, "show_hint_if_necessary",
            lambda prefs, hint, **kwargs: hint_calls.append((prefs, hint)),
        )
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert hint_calls == []


def preserve_tts_and_sgl_state():
    return {
        "had_tts_type": hasattr(Tts, "_type"),
        "tts_type": getattr(Tts, "_type", None),
        "backend_mode": getattr(Tts, "_backend_mode", None),
        "selected_server_model_id": Tts._selected_server_model_id,
        "remote_issue": Tts._remote_issue,
        "configured_definitions": Tts._configured_definitions,
    }

def restore_tts_and_sgl_state(saved) -> None:
    if saved["had_tts_type"]:
        Tts._type = saved["tts_type"]
    else:
        delattr(Tts, "_type")
    Tts._backend_mode = saved["backend_mode"]
    Tts._selected_server_model_id = saved["selected_server_model_id"]
    Tts._remote_issue = saved["remote_issue"]
    Tts._configured_definitions = saved["configured_definitions"]


def install_configured_definitions() -> None:
    """Simulate the SGL-Omni startup state: definitions always loaded."""
    from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
    Tts._configured_definitions = load_definitions().models


def test_tts_model_heading_detail_adds_sgl_omni_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.require_by_id("higgs_v3_sglomni")
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        install_configured_definitions()
        Tts._selected_server_model_id = "bosonai/higgs-audio-v3"
        monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
            backend_kind=TtsBackendKind.SGL_OMNI))

        result = get_heading_tts_text(make_state(TtsModelType.require_by_id("higgs_v3_sglomni")))

        assert text_util.strip_ansi_codes(result) == "Higgs Audio V3 SGL-Omni server model id: bosonai/higgs-audio-v3"
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_adds_offline_for_sgl_omni_without_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.require_by_id("higgs_v3_sglomni")
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        install_configured_definitions()
        Tts._selected_server_model_id = ""
        Tts._remote_issue = ""
        monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
            backend_kind=TtsBackendKind.SGL_OMNI,
            issue=RemoteTtsIssue("unavailable", "Remote server unavailable")))

        result = get_heading_tts_text(make_state(TtsModelType.require_by_id("higgs_v3_sglomni")))

        assert text_util.strip_ansi_codes(result) == "Higgs Audio V3 SGL-Omni: Remote server unavailable"
        assert COL_ERROR in result
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_keeps_local_model_unchanged():
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.require_by_id("chatterbox_local")
        Tts._backend_mode = TtsRuntimeMode.LOCAL
        Tts._selected_server_model_id = "bosonai/higgs-audio-v3"
        state = make_state(TtsModelType.require_by_id("chatterbox_local"))

        result = get_heading_tts_text(state)

        assert result == Tts.get_model_support(state.project).get_menu_text(state.project, None)
        assert "bosonai/higgs-audio-v3" not in text_util.strip_ansi_codes(result)
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_reads_cached_selection_without_polling(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.require_by_id("moss_delay_sglomni")
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
        install_configured_definitions()
        Tts._selected_server_model_id = "served/moss-delay"
        monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
            backend_kind=TtsBackendKind.SGL_OMNI))
        monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("heading must not perform network discovery")))

        result = get_heading_tts_text(make_state(TtsModelType.require_by_id("moss_delay_sglomni")))

        assert text_util.strip_ansi_codes(result) == "MOSS-TTS Delay SGL-Omni server model id: served/moss-delay"
    finally:
        restore_tts_and_sgl_state(saved)
