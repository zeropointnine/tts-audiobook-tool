from __future__ import annotations

from tts_audiobook_tool import app_support, text_util
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *


class MenuStatus:
    """
    Prints "menu status block" at top of console when in "clear screen mode"
    """

    @staticmethod
    def print_block(state: State) -> None:
        from tts_audiobook_tool.tts import Tts

        lines = []

        project_text = _make_project_text(state)
        lines.append(("Project", project_text))

        if Tts.get_type().value.is_sgl_omni or Tts.get_type() == TtsModelType.NONE:
            server_tts_text = _make_server_tts_text(state)
            lines.append(("SGL-Omni", server_tts_text))
        else:
            local_tts_text = _make_local_tts_text(state)        
            lines.append(("TTS model", local_tts_text))

        voice_text = _make_voice_text(state)
        lines.append(("Voice clone", voice_text))
        
        text_text = _make_text_text(state)
        lines.append(("Text", text_text))

        stt_text = _make_stt_text(state)
        lines.append(("STT model", stt_text))
        
        memory_text = _make_memory_text()
        if memory_text:
            lines.append(("Memory", memory_text))

        label_len = 0
        for label, _ in lines:
            label_len = max(label_len, len(label))

        for label, value in lines:
            label = (label + ":").ljust(label_len + 1)
            s = f"{LABEL_COLOR}{label} {VALUE_COLOR}{value}"
            printt(s)


def _make_project_text(state: State) -> str:
    if state.project.dir_path:
        text = text_util.make_terminal_hyperlink(state.project.dir_path)
    else:
        text = "none"

    language_code = state.project.language_code.strip()
    if state.project.dir_path and language_code:
        text += f" {QUALIFIER_COLOR}({language_code})"

    return text

def _make_local_tts_text(state: State) -> str:
    """ Eg: Some TTS (special sauce: True) (cuda, loaded)"""
    
    from tts_audiobook_tool.tts import Tts

    instance = Tts.get_instance_if_exists()
    text = Tts.get_class().get_menu_text(state.project, instance)
    
    extras = []
    if instance and instance.get_device():
        extras.append(instance.get_device())
    if instance:
        extras.append("loaded")
    if not instance and state.prefs.tts_force_cpu:
        # Note, showing "force cpu" qualifier only if no instance exists
        extras.append("will force cpu")

    if extras:
        text += f" {QUALIFIER_COLOR}({', '.join(extras)})"
    return text

def _make_server_tts_text(state: State) -> str:
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
    
    return f"{label}" + (f" {qualifier}" if qualifier else "")

def _make_voice_text(state: State) -> str:
    from tts_audiobook_tool.tts import Tts

    voice_prefix, voice_value = Tts.get_class().get_voice_display_info(
        state.project,
        Tts.get_instance_if_exists()
    )
    voice_prefix = text_util.strip_ansi_codes(voice_prefix).strip().rstrip(":")
    voice_value = text_util.strip_ansi_codes(voice_value).strip()
    return voice_value or voice_prefix or "none"

def _make_text_text(state: State) -> str:
    total_lines = len(state.project.phrase_groups)
    num_complete = state.project.sound_segments.num_generated()
    text = f"{total_lines} lines"
    text += f" {COL_DIM}({num_complete} generated)"
    return text


def _make_stt_text(state: State) -> str:
    from tts_audiobook_tool.stt import Stt

    text = "mlx-whisper" if Stt.should_use_mlx_whisper() else "faster-whisper"
    if state.prefs.stt_variant == SttVariant.DISABLED:
        text += " - disabled" # not dim

    qualifiers = []
    if not Stt.should_use_mlx_whisper():
        qualifiers.append(Stt.get_variant().id) # eg, "large-v3"
        qualifiers.append(state.prefs.stt_config.device) # eg, "cuda"
    if Stt.has_instance():
        qualifiers.append("loaded")
    if qualifiers:
        s2 = ", ".join(qualifiers)
        text += f" {COL_DIM}({s2})"
    
    return text

def _make_memory_text() -> str:
    text = text_util.strip_ansi_codes(app_support.make_memory_string())
    text = text.replace(":", "") # careful
    return text if text else ""

# ---

LABEL_COLOR = COL_DIM
VALUE_COLOR = COL_MEDIUM
QUALIFIER_COLOR = COL_DIM
