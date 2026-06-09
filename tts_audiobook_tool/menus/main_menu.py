from tts_audiobook_tool import text_util
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.menus.chat_menu import ChatMenu
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil, should_show_menu_status_details
from tts_audiobook_tool.menus.real_time_playback_menu import RealTimePlaybackMenu
from tts_audiobook_tool.menus.options_menu import OptionsMenu
from tts_audiobook_tool.menus.generate_menu import GenerateMenu
from tts_audiobook_tool.menus.project_menu import ProjectMenu
from tts_audiobook_tool.menus.tools_menu import ToolsMenu
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.menus.text_menu import TextMenu
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared


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
                MenuItem(
                    make_project_label, lambda _, __: ProjectMenu.menu(state), hotkey="p",
                    superlabel="Main", superlabel_no_blank_line=True
                )
            )
            items.append(
                MenuItem(make_voice_label, on_voice, hotkey="v")
            )
            items.append(
                MenuItem(make_text_label, on_text, hotkey="t")
            )
            items.append(
                MenuItem(
                    "Generate audio segments", on_generate, hotkey="g"
                )
            )
            items.append(
                MenuItem("Create audiobook file", on_concat, hotkey="c")
            )

            items.append(
                MenuItem(
                    "Realtime playback", on_realtime_audiobook, hotkey="r",
                    superlabel="Voicelab"
                )
            )
            items.append(
                MenuItem("LLM voice chat", on_chat, hotkey="l")
            )
            items.append(
                MenuItem(
                    "Options", lambda _, __: OptionsMenu.menu(state), hotkey="o",
                    superlabel="Options and tools"                    
                )
            )
            items.append(
                MenuItem(
                    "Tools", lambda _, __: ToolsMenu.menu(state), hotkey="z"
                )
            )
            items.append(
                MenuItem("Quit", on_quit, hotkey="q")
            )
            return items

        if should_show_menu_status_details(state):
            s = make_tts_model_heading_detail(state)
            heading = f"{text_util.make_terminal_hyperlink(APP_URL, APP_NAME)} {COL_DIM}(TTS model: {COL_ACCENT}{s}{COL_DIM})"
        else:
            heading = f"{text_util.make_terminal_hyperlink(APP_URL, APP_NAME)}"
        
        MenuUtil.menu(state, heading, make_items, is_submenu=False, one_shot=True, breadcrumb="Main")

# ---

def make_tts_model_heading_detail(state: State) -> str:
    s = Tts.get_class().get_model_display_text(state.project, Tts.get_instance_if_exists())
    if not Tts.is_sgl_mode():
        return s
    SglOmniUtil.update_model_id()
    if SglOmniUtil.get_model_id():
        s += f" {COL_DIM}server model id: {SglOmniUtil.get_model_id()}{COL_ACCENT}"
    else:
        s += f" {COL_ERROR}SGL-Omni offline{COL_ACCENT}"
    return s

# ---

# Project
def make_project_label(state: State) -> str:
    if should_show_menu_status_details(state):
        currently = make_currently_string(
            state.project.dir_path, 
            required_predicate=lambda: not bool(state.project.dir_path),
            required_label="required - set this first"
        )
    else:
        currently = ""

    start_here = f" {COL_ERROR}<-- start here" if not state.project.dir_path else ""
    return f"Project {currently}{start_here}"

# Voice
def make_voice_label(state: State) -> str:
    base_label = f"Voice clone and model settings"
    if not should_show_menu_status_details(state):
        return base_label
    prefix, value = Tts.get_class().get_voice_display_info(state.project, Tts.get_instance_if_exists())
    if not value:
        combined = prefix
    else:
        combined = f"{prefix}: {value}"
    
    return f"{base_label} {COL_DIM}({combined}{COL_DIM})"

def on_voice(state: State, __) -> None:
    Tts.update_tts_type()
    if Tts.get_type() == TtsModelType.NONE:
        print_feedback("Requires TTS model", is_error=True)
        return
    if not state.project.dir_path:
        print_feedback(REQUIRES_PROJECT, is_error=True)
        return
    VoiceMenuShared.menu(state)

# Text
def make_text_label(state: State) -> str:
    if state.project.phrase_groups:
        if state.prefs.menu_clears_screen:
            current = ""
        else:
            num = len(state.project.phrase_groups)
            lines = make_noun("line", "lines", num)
            current = make_currently_string(f"{num} {lines}")
    else:
        current = ""
    return f"Text {current}"

def on_text(state: State, __) -> None:
    if not state.project.dir_path:
        print_feedback(REQUIRES_PROJECT, is_error=True)
        return
    TextMenu.menu(state)

def on_generate(state: State, _: MenuItem) -> None:
    if not state.project.dir_path:
        print_feedback(REQUIRES_PROJECT, is_error=True)
        return
    GenerateMenu.menu(state)

def on_concat(state: State, _: MenuItem) -> None:
    if not state.project.dir_path:
        print_feedback(REQUIRES_PROJECT, is_error=True)
        return
    ConcatMenu.menu(state)

def on_realtime_audiobook(state: State, _: MenuItem) -> None:
    if not state.project.dir_path:
        print_feedback(REQUIRES_PROJECT, is_error=True)
        return
    RealTimePlaybackMenu.menu(state)

def on_chat(state: State, _: MenuItem) -> None:
    Tts.update_tts_type()
    if Tts.get_type() == TtsModelType.NONE:
        print_feedback(REQUIRES_TTS_MODEL, is_error=True)
        return
    if not state.project.dir_path:
        print_feedback(REQUIRES_PROJECT, is_error=True)
        return
    ChatMenu.menu(state)

# Quit
def on_quit(_: State, __: MenuItem):
    print_feedback("State saved.", extra_line=False, skip_pause=True)
    exit(0)

REQUIRES_PROJECT = "Requires a project"
REQUIRES_TTS_MODEL = "Requires TTS model"
