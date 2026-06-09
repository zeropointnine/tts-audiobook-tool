from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.menus.main_menu import make_tts_model_heading_detail
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_ERROR


def make_state() -> State:
    state = object.__new__(State)
    state._prefs = Prefs()
    state._project = Project(dir_path="")
    return state


def preserve_tts_and_sgl_state():
    return {
        "had_tts_type": hasattr(Tts, "_type"),
        "tts_type": getattr(Tts, "_type", None),
        "sgl_omni_type": Tts._sgl_omni_type,
        "base_url": SglOmniUtil._base_url,
        "model_id": SglOmniUtil._model_id,
    }

def restore_tts_and_sgl_state(saved) -> None:
    if saved["had_tts_type"]:
        Tts._type = saved["tts_type"]
    else:
        delattr(Tts, "_type")
    Tts._sgl_omni_type = saved["sgl_omni_type"]
    SglOmniUtil._base_url = saved["base_url"]
    SglOmniUtil._model_id = saved["model_id"]


def test_tts_model_heading_detail_adds_sgl_omni_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.HIGGS_V3_SERVER
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3"
        monkeypatch.setattr(SglOmniUtil, "update_model_id", lambda: None)

        result = make_tts_model_heading_detail(make_state())

        assert text_util.strip_ansi_codes(result) == "Higgs Audio V3 SGL-Omni model id: bosonai/higgs-audio-v3"
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_adds_offline_for_sgl_omni_without_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.HIGGS_V3_SERVER
        SglOmniUtil._model_id = ""
        monkeypatch.setattr(SglOmniUtil, "update_model_id", lambda: None)

        result = make_tts_model_heading_detail(make_state())

        assert text_util.strip_ansi_codes(result) == "Higgs Audio V3 offline"
        assert COL_ERROR in result
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_keeps_local_model_unchanged():
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.CHATTERBOX
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3"
        state = make_state()

        result = make_tts_model_heading_detail(state)

        assert result == Tts.get_class().get_model_display_text(state.project, Tts.get_instance_if_exists())
        assert "bosonai/higgs-audio-v3" not in text_util.strip_ansi_codes(result)
    finally:
        restore_tts_and_sgl_state(saved)


def test_tts_model_heading_detail_refreshes_stale_sgl_omni_model_id(monkeypatch):
    saved = preserve_tts_and_sgl_state()
    try:
        Tts._type = TtsModelType.MOSS_SERVER
        SglOmniUtil._model_id = "bosonai/higgs-audio-v3-tts-4b"

        def update_model_id():
            SglOmniUtil._model_id = ""

        monkeypatch.setattr(SglOmniUtil, "update_model_id", update_model_id)

        result = make_tts_model_heading_detail(make_state())

        stripped = text_util.strip_ansi_codes(result)
        assert stripped == "MOSS-TTS offline"
        assert "bosonai/higgs-audio-v3-tts-4b" not in stripped
        assert COL_ERROR in result
    finally:
        restore_tts_and_sgl_state(saved)
