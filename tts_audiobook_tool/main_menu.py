from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.real_time_submenu import RealTimeSubmenu
from tts_audiobook_tool.options_submenu import OptionsSubmenu
from tts_audiobook_tool.generate_submenu import GenerateSubmenu
from tts_audiobook_tool.project_submenu import ProjectSubmenu
from tts_audiobook_tool.tools_submenu import ToolsSubmenu
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.text_submenu import TextSubmenu
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State

class MainMenu:
    """
    Main menu and misc submenus
    """

    @staticmethod
    def menu_loop(state: State) -> None:

        while True:

            if state.prefs.project_dir and not os.path.exists(state.prefs.project_dir):
                printt(f"{COL_ERROR}Project directory {state.prefs.project_dir} not found.")
                printt()
                state.reset()

            MainMenu.menu(state)

    @staticmethod
    def menu(state: State) -> None:

        def make_menu_items(_) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(make_project_label, lambda _, __: ProjectSubmenu.menu(state), hotkey="p")
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
                    MenuItem(make_concat_label, on_concat, hotkey="c")
                )
            if state.prefs.project_dir and Tts.get_type() != TtsModelInfos.NONE:
                items.append(
                    MenuItem(make_realtime_label, on_realtime, hotkey="r")
                )
            items.append(
                MenuItem("Tools", lambda _, __: ToolsSubmenu.menu(state), hotkey="z")
            )
            items.append(
                MenuItem("Options", lambda _, __: OptionsSubmenu.menu(state), hotkey="o")
            )
            items.append(
                MenuItem("Quit", on_quit, hotkey="q")
            )
            return items

        heading = f"{APP_NAME} {COL_DIM}(active model: {COL_ACCENT}{Tts.get_type().value.ui['proper_name']}{COL_DIM})"
        MenuUtil.menu(state, heading, make_menu_items, is_submenu=False, one_shot=True)

# ---

# Project
def make_project_label(state: State) -> str:
    if not state.prefs.project_dir:
        s = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
    else:
        s = make_currently_string(state.prefs.project_dir)
    return "Project directory " + s

# Voice
def make_voice_label(state: State) -> str:
    voice_label = state.project.get_voice_label()
    if not state.project.can_voice:
        current = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
    else:
        if voice_label != "none":
            current = make_currently_string(voice_label, label="current voice clone: ")
        else:
            current = make_currently_string(voice_label, label="current voice clone: ", color_code=COL_ERROR)
    return f"Voice clone and model settings {current}"

def on_voice(state: State, __) -> None:
    if not state.prefs.project_dir:
        return
    match Tts.get_type():
        case TtsModelInfos.OUTE:
            from tts_audiobook_tool.voice_oute_submenu import VoiceOuteSubmenu
            VoiceOuteSubmenu.menu(state)
        case TtsModelInfos.CHATTERBOX:
            from tts_audiobook_tool.voice_chatterbox_submenu import VoiceChatterboxSubmenu
            VoiceChatterboxSubmenu.menu(state)
        case TtsModelInfos.FISH:
            from tts_audiobook_tool.voice_fish_submenu import VoiceFishSubmenu
            VoiceFishSubmenu.menu(state)
        case TtsModelInfos.HIGGS:
            from tts_audiobook_tool.voice_higgs_submenu import VoiceHiggsSubmenu
            VoiceHiggsSubmenu.menu(state)
        case TtsModelInfos.VIBEVOICE:
            from tts_audiobook_tool.voice_vibevoice_submenu import VoiceVibeVoiceSubmenu
            VoiceVibeVoiceSubmenu.menu(state)
        case TtsModelInfos.INDEXTTS2:
            from tts_audiobook_tool.voice_indextts2_submenu import VoiceIndexTts2Submenu
            VoiceIndexTts2Submenu.menu(state)

# Text
def make_text_label(state: State) -> str:
    if state.project.text_segments:
        lines = "line" if len(state.project.text_segments) == 1 else "lines"
        value = f"{len(state.project.text_segments)} {lines}"
        current = make_currently_string(value)
    else:
        current = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
    return f"Text {current}"

def on_text(state: State, __) -> None:
    if not state.prefs.project_dir:
        print_feedback("Requires project", is_error=True)
        return
    if not state.project.text_segments:
        TextSubmenu.set_text_menu(state)
    else:
        TextSubmenu.replace_text_menu(state)

# Gen
def make_gen_label(state: State) -> str:
    num_generated = state.project.sound_segments.num_generated()
    if not state.project.can_voice and not state.project.text_segments:
        s = f"{COL_DIM}({COL_ERROR}requires text and voice sample{COL_DIM})"
    elif not state.project.can_voice:
        s = f"{COL_DIM}({COL_ERROR}requires voice sample{COL_DIM})"
    elif not state.project.text_segments:
        s = f"{COL_DIM}({COL_ERROR}requires text{COL_DIM})"
    else:
        s = f"{COL_DIM}({COL_ACCENT}{num_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} lines complete)"
    return "Generate audiobook audio " + s

def on_gen(state: State, __) -> None:
    if not state.project.can_voice:
        print_feedback("Requires voice clone sample", is_error=True)
        return
    if len(state.project.text_segments) == 0:
        print_feedback("Requires text", is_error=True)
        return
    GenerateSubmenu.menu(state)

# Concat
def make_concat_label(state: State) -> str:
    num_generated = state.project.sound_segments.num_generated()
    s = "Concatenate audiobook lines to create audiobook file "
    if num_generated == 0:
        s += f"{COL_DIM}({COL_ERROR}requires generated audio{COL_DIM})"
    return s

def on_concat(state: State, __) -> None:
    num_generated = state.project.sound_segments.num_generated()
    if not state.prefs.project_dir or num_generated == 0:
        return
    ConcatSubmenu.menu(state)

# Realtime
def make_realtime_label(state: State) -> str:
    if not state.project.can_voice and not state.project.text_segments:
        s = f"{COL_DIM}({COL_ERROR}requires text and voice sample{COL_DIM})"
    elif not state.project.can_voice:
        s = f"{COL_DIM}({COL_ERROR}requires voice sample{COL_DIM})"
    elif not state.project.text_segments:
        s = f"{COL_DIM}({COL_ERROR}requires text{COL_DIM})"
    else:
        s = ""
    return "Generate realtime audio " + s

def on_realtime(state: State, __) -> None:
    if not state.project.text_segments:
        print_feedback("Requires text", is_error=True)
        return
    if not state.project.can_voice:
        print_feedback("Requires voice clone sample", is_error=True)
        return
    RealTimeSubmenu.menu(state)

# Quit
def on_quit(_, __):
    print_feedback("State saved.", extra_line=False)
    exit(0)
