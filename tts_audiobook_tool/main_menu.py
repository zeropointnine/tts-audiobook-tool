from tts_audiobook_tool.concat_menu import ConcatMenu
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.real_time_menu import RealTimeMenu
from tts_audiobook_tool.options_menu import OptionsMenu
from tts_audiobook_tool.generate_menu import GenerateMenu
from tts_audiobook_tool.project_menu import ProjectMenu
from tts_audiobook_tool.tools_menu import ToolsMenu
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.text_menu import TextMenu
from tts_audiobook_tool.tts_model import Qwen3Protocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_menu import VoiceQwen3Menu


class MainMenu:
    """
    """

    @staticmethod
    def menu_loop(state: State) -> None:
        """
        This acts as the main program loop.
        """

        while True:

            if state.prefs.project_dir and not os.path.exists(state.prefs.project_dir):
                printt(f"{COL_ERROR}Project directory {state.prefs.project_dir} not found.")
                printt()
                state.reset()

            MainMenu.menu(state)

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(make_project_label, lambda _, __: ProjectMenu.menu(state), hotkey="p")
            )
            if state.prefs.project_dir and Tts.get_type() != TtsModelInfos.NONE:
                items.append(
                    MenuItem(make_voice_label, on_voice, hotkey="v")
                )
            if state.prefs.project_dir:
                items.append(
                    MenuItem(make_text_label, on_text, hotkey="t")
                )
            if state.prefs.project_dir and Tts.get_type() != TtsModelInfos.NONE:
                items.append(
                    MenuItem(make_gen_label, on_gen, hotkey="g")
                )
            if state.prefs.project_dir:
                items.append(
                    MenuItem(make_concat_label, lambda _, __: ConcatMenu.menu(state), hotkey="c")
                )
            if state.prefs.project_dir and Tts.get_type() != TtsModelInfos.NONE:
                items.append(
                    MenuItem(make_realtime_label, on_realtime, hotkey="r")
                )
            items.append(
                MenuItem("Tools", lambda _, __: ToolsMenu.menu(state), hotkey="z")
            )
            items.append(
                MenuItem("Options", lambda _, __: OptionsMenu.menu(state), hotkey="o")
            )
            items.append(
                MenuItem("Quit", on_quit, hotkey="q")
            )
            return items

        if Tts.get_type() == TtsModelInfos.CHATTERBOX:
            model_name = state.project.chatterbox_type.label
        else:
            model_name = Tts.get_type().value.ui['proper_name']
        if Tts.get_type() == TtsModelInfos.QWEN3TTS:
            path = state.project.qwen3_target or Qwen3Protocol.DEFAULT_REPO_ID
            s = truncate_path_pretty(path)
            model_name += f" {COL_DIM}{s}"

        heading = f"{APP_NAME} {COL_DIM}(active model: {COL_ACCENT}{model_name}{COL_DIM})"
        
        MenuUtil.menu(state, heading, make_items, is_submenu=False, one_shot=True)

# ---

# Project
def make_project_label(state: State) -> str:
    if not state.prefs.project_dir:
        s = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
    else:
        s = make_currently_string(state.prefs.project_dir)
    return "Project " + s

# Voice
def make_voice_label(state: State) -> str:

    voice_label = state.project.get_voice_label()

    if Tts.get_type() == TtsModelInfos.VIBEVOICE:

        lora_label = state.project.vibevoice_lora_target
        if voice_label == "none" and not lora_label:
            desc = make_currently_string(voice_label, value_prefix="current voice clone: ", color_code=COL_ERROR)
        elif voice_label != "none" and not lora_label:
            desc = make_currently_string(voice_label, value_prefix="current voice clone: ")
        elif voice_label == "none" and lora_label:
            desc = make_currently_string(lora_label, value_prefix="current LoRA: ")
        else: # has both
            desc = make_currently_string(f"{voice_label}, {lora_label}", value_prefix="current voice clone and LoRA: ")

    elif Tts.get_type() == TtsModelInfos.QWEN3TTS:
        
        if not Tts.instance_exists():
            # Model not loaded, and we don't want to make loaded model a requirement for main menu
            desc = ""
        else:
            desc = Tts.get_qwen3().make_main_menu_model_desc(state.project)

    else:

        voice_label = state.project.get_voice_label()
        if not state.project.can_voice:
            desc = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
        else:
            if voice_label != "none":
                desc = make_currently_string(voice_label, value_prefix="current voice clone: ")
            else:
                desc = make_currently_string(voice_label, value_prefix="current voice clone: ", color_code=COL_ERROR)
    
    return f"Voice clone and model settings {desc}"

def on_voice(state: State, __) -> None:
    if not state.prefs.project_dir:
        return
    match Tts.get_type():
        case TtsModelInfos.OUTE:
            from tts_audiobook_tool.voice_menu import VoiceOuteMenu
            VoiceOuteMenu.menu(state)
        case TtsModelInfos.CHATTERBOX:
            from tts_audiobook_tool.voice_menu import VoiceChatterboxMenu
            VoiceChatterboxMenu.menu(state)
        case TtsModelInfos.FISH:
            from tts_audiobook_tool.voice_menu import VoiceFishMenu
            VoiceFishMenu.menu(state)
        case TtsModelInfos.HIGGS:
            from tts_audiobook_tool.voice_menu import VoiceHiggsMenu
            VoiceHiggsMenu.menu(state)
        case TtsModelInfos.VIBEVOICE:
            from tts_audiobook_tool.voice_menu import VoiceVibeVoiceMenu
            VoiceVibeVoiceMenu.menu(state)
        case TtsModelInfos.INDEXTTS2:
            from tts_audiobook_tool.voice_menu import VoiceIndexTts2Menu
            VoiceIndexTts2Menu.menu(state)
        case TtsModelInfos.GLM:
            from tts_audiobook_tool.voice_menu import VoiceGlmMenu
            VoiceGlmMenu.menu(state)
        case TtsModelInfos.MIRA:
            from tts_audiobook_tool.voice_menu import VoiceMiraMenu
            VoiceMiraMenu.menu(state)
        case TtsModelInfos.QWEN3TTS:
            _ = Tts.get_instance() # Pre-emptively instantiate model (special case for qwen)
            VoiceQwen3Menu.menu(state)
        case _:
            ...

# Text
def make_text_label(state: State) -> str:
    if state.project.phrase_groups:
        num = len(state.project.phrase_groups)
        lines = make_noun("line", "lines", num)
        current = make_currently_string(f"{num} {lines}")
    else:
        current = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
    return f"Text {current}"

def on_text(state: State, __) -> None:
    if not state.prefs.project_dir:
        print_feedback("Requires project", is_error=True)
        return
    TextMenu.menu(state)

# Gen
def make_gen_label(state: State) -> str:
    num_generated = state.project.sound_segments.num_generated()
    if not state.project.can_voice and not state.project.phrase_groups:
        s = f"{COL_DIM}({COL_ERROR}requires text and voice sample{COL_DIM})"
    elif not state.project.can_voice:
        s = f"{COL_DIM}({COL_ERROR}requires voice sample{COL_DIM})"
    elif not state.project.phrase_groups:
        s = f"{COL_DIM}({COL_ERROR}requires text{COL_DIM})"
    else:
        s = f"{COL_DIM}({COL_ACCENT}{num_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.phrase_groups)}{COL_DIM} lines complete)"
    return "Generate audio " + s

def on_gen(state: State, __) -> None:
    if not state.project.can_voice:
        print_feedback("Requires voice clone sample", is_error=True)
        return
    if len(state.project.phrase_groups) == 0:
        print_feedback("Requires text", is_error=True)
        return
    GenerateMenu.menu(state)

# Concat
def make_concat_label(state: State) -> str:
    num_generated = state.project.sound_segments.num_generated()
    s = "Concatenate audiobook lines to create audiobook file "
    if num_generated == 0:
        s += f"{COL_DIM}({COL_ERROR}requires generated audio{COL_DIM})"
    return s

# Realtime
def make_realtime_label(state: State) -> str:
    s = "Realtime audio generation "
    if not state.project.can_voice and not state.project.phrase_groups:
        s += f"{COL_DIM}({COL_ERROR}requires text and voice sample{COL_DIM})"
    elif not state.project.can_voice:
        s += f"{COL_DIM}({COL_ERROR}requires voice sample{COL_DIM})"
    elif not state.project.phrase_groups:
        s += f"{COL_DIM}({COL_ERROR}requires text{COL_DIM})"
    else:
        ...
    return s

def on_realtime(state: State, __) -> None:
    if not state.project.phrase_groups:
        print_feedback("Requires text", is_error=True)
        return
    if not state.project.can_voice:
        print_feedback("Requires voice clone sample", is_error=True)
        return
    RealTimeMenu.menu(state)

# Quit
def on_quit(_: State, __: MenuItem):
    print_feedback("State saved.", extra_line=False)
    exit(0)
