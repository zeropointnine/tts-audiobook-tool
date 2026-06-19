from __future__ import annotations

from tts_audiobook_tool import app_support, text_util
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *


class MenuStatus:
    """
    Prints "menu status block" at top of console
    """

    @staticmethod
    def print_block(state: State) -> None:
        from tts_audiobook_tool.tts import Tts

        project_line = _make_project_line(state)
        printt(f"{LABEL_COLOR}Project:     {project_line}")

        if Tts.get_type().value.is_sgl_omni or Tts.get_type() == TtsModelType.NONE:
            sgl_omni_line = _make_sgl_omni_line(state)
            printt(f"{LABEL_COLOR}SGL-Omni:    {sgl_omni_line}")
        else:
            tts_line = _make_tts_line(state)        
            printt(f"{LABEL_COLOR}TTS model:   {tts_line}")

        voice_line = _make_voice_line(state)
        printt(f"{LABEL_COLOR}Voice clone: {voice_line}")
        
        text_line = _make_text_line(state)
        printt(f"{LABEL_COLOR}Text:        {text_line}")

        stt_line = _make_stt_line(state)
        printt(f"{LABEL_COLOR}STT model:   {stt_line}")
        
        memory_line = _make_memory_line()        
        if memory_line:
            printt(f"{LABEL_COLOR}Memory:      {memory_line}")


def _make_project_line(state: State) -> str:
    if state.project.dir_path:
        project_line = VALUE_COLOR + text_util.make_terminal_hyperlink(state.project.dir_path)
    else:
        project_line = VALUE_COLOR + "none"

    language_code = state.project.language_code.strip()
    if state.project.dir_path and language_code:
        project_line += f" {QUALIFIER_COLOR}({language_code})"

    return project_line

def _make_tts_line(state: State) -> str:
    from tts_audiobook_tool.tts import Tts

    if Tts._instance_display_info:
        tts_line = VALUE_COLOR + Tts._instance_display_info.model_description
        match (bool(Tts._instance_display_info.device), bool(Tts._instance_display_info.extra)):
            case (True, True):
                extra = f"{Tts._instance_display_info.device}, {Tts._instance_display_info.extra}"
            case (True, False):
                extra = Tts._instance_display_info.device
            case (False, True):
                extra = Tts._instance_display_info.extra
            case (False, False):
                extra = ""
        if extra:
            extra = f"({extra})"
        extra += " (loaded)"
        tts_line += f" {QUALIFIER_COLOR}{extra}"
    else:
        tts_line = VALUE_COLOR + Tts.get_class().INFO.ui['proper_name']
        if state.prefs.tts_force_cpu:
            # Note, showing "force cpu" qualifier only if no instance exists
            # b/c if instance exists, will already show device value cpus
            tts_line += f" {QUALIFIER_COLOR}(force cpu)"

    return tts_line

def _make_sgl_omni_line(state: State) -> str:
    from tts_audiobook_tool.tts import Tts
    
    if Tts.get_type() == TtsModelType.NONE:
        if SglOmniUtil.get_model_id():
            label = f"{COL_ERROR}Unknown/unsupported"
            model_id = ellipsize(SglOmniUtil.get_model_id(), 40, from_start=True)
            qualifier = f"{QUALIFIER_COLOR}({model_id})"
        else:                
            label = f"{COL_ERROR}Offline"
            url = ellipsize(SglOmniUtil.get_base_url(), 40)
            qualifier = f"{QUALIFIER_COLOR}({url})"
    else:
        label = Tts.get_type().value.ui["proper_name"]
        if state.prefs.sgl_omni_type is None:
            model_id = ellipsize(SglOmniUtil.get_model_id(), 40, from_start=True)
            qualifier = f"{QUALIFIER_COLOR}({model_id})" if model_id else ""
        else:
            if not SglOmniUtil.get_model_id():
                SglOmniUtil.update_model_id()
            if SglOmniUtil.get_model_id():
                # When sgl tts type is set explicitly, we want to draw extra attention to the server model id,
                # hence the extra qualifier
                model_id = ellipsize(SglOmniUtil.get_model_id(), 40, from_start=True)
                qualifier = f"{QUALIFIER_COLOR}(server model id: {model_id})"
            else:
                qualifier = f"{COL_ERROR}(offline)"
    
    return f"{VALUE_COLOR}{label}" + (f" {qualifier}" if qualifier else "")

def _make_voice_line(state: State) -> str:
    from tts_audiobook_tool.tts import Tts

    voice_prefix, voice_value = Tts.get_class().get_voice_display_info(
        state.project,
        Tts.get_instance_if_exists()
    )
    voice_prefix = text_util.strip_ansi_codes(voice_prefix).strip().rstrip(":")
    voice_value = text_util.strip_ansi_codes(voice_value).strip()
    return VALUE_COLOR + (voice_value or voice_prefix or "none")


def _make_text_line(state: State) -> str:
    total_lines = len(state.project.phrase_groups)
    num_complete = state.project.sound_segments.num_generated()
    text_line = VALUE_COLOR + f"{total_lines} lines"
    text_line += f" {COL_DIM}({num_complete} generated)"
    return text_line


def _make_stt_line(state: State) -> str:
    from tts_audiobook_tool.stt import Stt

    stt_model = "mlx-whisper" if Stt.should_use_mlx_whisper() else "faster-whisper"
    stt_line = VALUE_COLOR + stt_model
    if state.prefs.stt_variant == SttVariant.DISABLED:
        stt_line += " disabled" # not dim
    else:
        if Stt.has_instance():
            stt_variant = Stt.get_variant().id
            fw_config = state.prefs.stt_config.description if not Stt.should_use_mlx_whisper() else ""
            stt_line += f" {stt_variant} {COL_DIM}({fw_config}) {COL_DIM}(loaded)"
        else:
            stt_line += ""

    return stt_line


def _make_memory_line() -> str:
    memory_line = text_util.strip_ansi_codes(app_support.make_memory_string())
    memory_line = memory_line.replace(":", "") # careful
    return VALUE_COLOR + memory_line if memory_line else ""


LABEL_COLOR = COL_DIM
VALUE_COLOR = COL_MEDIUM
QUALIFIER_COLOR = COL_DIM
