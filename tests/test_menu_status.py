from types import SimpleNamespace
from typing import cast

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.menus.main_menu import make_voice_label
from tts_audiobook_tool.menus.menu_status import _make_stt_text
from tts_audiobook_tool.menus.menu_status import MenuStatus
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool import text_util
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import COL_DIM, COL_ERROR, COL_MEDIUM


def make_state() -> State:
    state = object.__new__(State)
    state._prefs = Prefs()
    state._project = Project(dir_path="")
    return state


def preserve_tts_state():
    return {
        "had_tts_type": hasattr(Tts, "_type"),
        "tts_type": getattr(Tts, "_type", None),
    }


def restore_tts_state(saved) -> None:
    if saved["had_tts_type"]:
        Tts._type = saved["tts_type"]
    else:
        delattr(Tts, "_type")


def test_menu_status_print_block_supports_voice_display_info(capsys):
    saved = preserve_tts_state()
    try:
        Tts._type = TtsModelType.OMNIVOICE
        state = make_state()
        state.project.omnivoice_voice_file_name = [
            "zzz belle 24a 19s_omnivoice.flac",
            "zzz belle 24b 20s_omnivoice.flac",
        ]

        MenuStatus.print_block(state)

        output = capsys.readouterr().out
        assert f"{COL_DIM}Voice clone: {COL_MEDIUM}zzz belle 24a 19s, +1 more" in output
    finally:
        restore_tts_state(saved)


def test_menus_omit_absent_voice_display_info(monkeypatch, capsys):
    saved = preserve_tts_state()
    try:
        Tts._type = TtsModelType.OMNIVOICE
        state = make_state()
        tts_class = Tts.get_class()
        monkeypatch.setattr(
            tts_class,
            "get_voice_display_info",
            classmethod(lambda cls, project, instance=None: None),
        )

        MenuStatus.print_block(state)

        output = capsys.readouterr().out
        assert "Voice clone:" not in output
        assert make_voice_label(state) == "Voice clone and model settings"
    finally:
        restore_tts_state(saved)


def test_status_block_local_mode_none_shows_tts_model_line(capsys):
    # (local mode, NONE) is the "no model" state: it has no SGL-Omni surface
    saved = preserve_tts_state()
    try:
        Tts._type = TtsModelType.NONE
        Tts._backend_mode = TtsBackendKind.LOCAL
        state = make_state()

        MenuStatus.print_block(state)

        output = capsys.readouterr().out
        assert "SGL-Omni" not in output
        lines = [text_util.strip_ansi_codes(line) for line in output.splitlines()]
        lines = [line for line in lines if "TTS model:" in line]
        assert len(lines) == 1
        assert lines[0].strip().endswith("None")
    finally:
        restore_tts_state(saved)


def test_status_block_sgl_mode_none_shows_sgl_omni_line(capsys, monkeypatch):
    # (SGL-Omni mode, NONE) means "server not configured" and keeps the SGL-Omni line
    saved = preserve_tts_state()
    try:
        Tts._type = TtsModelType.NONE
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._base_url = "http://example.test"
        SglOmniUtil._model_id = ""
        monkeypatch.setattr(SglOmniUtil, "update_model_id", lambda: None)
        state = make_state()

        MenuStatus.print_block(state)

        output = capsys.readouterr().out
        lines = [text_util.strip_ansi_codes(line) for line in output.splitlines()]
        lines = [line for line in lines if "SGL-Omni:" in line]
        assert len(lines) == 1
        assert "Offline" in lines[0]
        assert "TTS model:" not in output
    finally:
        restore_tts_state(saved)
        SglOmniUtil._base_url = ""
        SglOmniUtil._model_id = ""


def test_dependent_voice_display_info_overrides_propagate_none(monkeypatch):
    project = Project(dir_path="")
    monkeypatch.setattr(
        TtsBaseModel,
        "get_voice_display_info",
        classmethod(lambda cls, project, instance=None: None),
    )

    assert Qwen3BaseModel.get_voice_display_info(project) is None
    assert IndexTts2BaseModel.get_voice_display_info(project) is None


def test_qwen_custom_voice_status_uses_project_speaker_without_instance():
    project = Project(dir_path="")
    project.qwen3_model_type = "custom_voice"
    project.qwen3_speaker_id = "Ryan"

    display_info = Qwen3BaseModel.get_voice_display_info(project, None)

    assert display_info is not None
    assert display_info.status_prefix == "Speaker"
    assert display_info.main_prefix == "speaker"
    assert display_info.value == "Ryan"


def test_qwen_custom_voice_status_includes_instructions_with_valid_speaker():
    project = Project(dir_path="")
    project.qwen3_model_type = "custom_voice"
    project.qwen3_speaker_id = "Ryan"
    project.qwen3_instructions = "Speak warmly"

    display_info = Qwen3BaseModel.get_voice_display_info(project, None)

    assert display_info is not None
    assert display_info.value == f"Ryan{COL_DIM} + instructions"


def test_qwen_custom_voice_status_requires_speaker_without_instance():
    project = Project(dir_path="")
    project.qwen3_model_type = "custom_voice"

    display_info = Qwen3BaseModel.get_voice_display_info(project, None)

    assert display_info is not None
    assert display_info.status_prefix == "Speaker"
    assert display_info.main_prefix == "speaker"
    assert display_info.value == COL_ERROR + "required"


def test_stt_status_does_not_repeat_disabled(monkeypatch):
    state = cast(State, SimpleNamespace(
        prefs=SimpleNamespace(
            stt_variant=SttVariant.DISABLED,
            stt_config=SttConfig.CPU_INT8FLOAT32,
        ),
    ))
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    monkeypatch.setattr(Stt, "get_variant", lambda: SttVariant.DISABLED)
    monkeypatch.setattr(Stt, "has_instance", lambda: False)

    text = _make_stt_text(state)

    assert text == f"faster-whisper {COL_DIM}(disabled)"
