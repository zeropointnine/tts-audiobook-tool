from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.constants_hints import HINT_SGL_OMNI_URL
from tts_audiobook_tool.menus.main_menu import MainMenu, get_heading_tts_text
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import COL_ERROR


def make_state() -> State:
    state = object.__new__(State)
    state._prefs = Prefs()
    state._project = Project(dir_path="")
    state.has_shown_main_menu = False
    return state


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
        Tts._backend_mode = TtsBackendKind.LOCAL
        state = make_state()
        marks = []
        state.mark_main_menu_shown = lambda: marks.append(True)  # type: ignore[method-assign]
        _capture_and_invoke_on_shown(monkeypatch, state)
    finally:
        restore_tts_and_sgl_state(saved)

    assert marks == [True]


def test_main_menu_on_shown_shows_sgl_omni_url_hint_when_offline_and_unset(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    state = None
    try:
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = ""
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
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = ""
        state = make_state()
        state._prefs = Prefs(sgl_omni_url="http://example.test:9009")
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
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3"
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


def test_main_menu_on_shown_does_not_show_sgl_omni_url_hint_in_local_mode(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    hint_calls = []
    try:
        Tts._backend_mode = TtsBackendKind.LOCAL
        SglOmniUtil._model_id = ""
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
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = ""
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
        "sgl_omni_type": Tts._sgl_omni_type,
        "base_url": SglOmniUtil._base_url,
        "model_id": SglOmniUtil._model_id,
    }

def restore_tts_and_sgl_state(saved) -> None:
    if saved["had_tts_type"]:
        Tts._type = saved["tts_type"]
    else:
        delattr(Tts, "_type")
    if saved["backend_mode"] is not None:
        Tts._backend_mode = saved["backend_mode"]
    Tts._sgl_omni_type = saved["sgl_omni_type"]
    SglOmniUtil._base_url = saved["base_url"]
    SglOmniUtil._model_id = saved["model_id"]


def test_tts_model_heading_detail_adds_sgl_omni_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.HIGGS_V3_SERVER
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3"
        monkeypatch.setattr(SglOmniUtil, "update_model_id", lambda: None)

        result = get_heading_tts_text(make_state())

        assert text_util.strip_ansi_codes(result) == "Higgs Audio V3 server model id: bosonai/higgs-audio-v3"
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_adds_offline_for_sgl_omni_without_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.HIGGS_V3_SERVER
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = ""
        monkeypatch.setattr(SglOmniUtil, "update_model_id", lambda: None)

        result = get_heading_tts_text(make_state())

        assert text_util.strip_ansi_codes(result) == "Higgs Audio V3 SGL-Omni offline"
        assert COL_ERROR in result
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_keeps_local_model_unchanged():
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.CHATTERBOX
        Tts._backend_mode = TtsBackendKind.LOCAL
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3"
        state = make_state()

        result = get_heading_tts_text(state)

        assert result == Tts.get_class().get_menu_text(state.project, Tts.get_instance_if_exists())
        assert "bosonai/higgs-audio-v3" not in text_util.strip_ansi_codes(result)
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_refreshes_stale_sgl_omni_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.MOSS_SERVER
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3-tts-4b"

        def update_model_id():
            SglOmniUtil._model_id = ""

        monkeypatch.setattr(SglOmniUtil, "update_model_id", update_model_id)

        result = get_heading_tts_text(make_state())

        stripped = text_util.strip_ansi_codes(result)
        assert stripped == "MOSS-TTS-v1.5 SGL-Omni offline"
        assert "bosonai/higgs-audio-v3-tts-4b" not in stripped
        assert COL_ERROR in result
    finally:
        restore_tts_and_sgl_state(saved)
