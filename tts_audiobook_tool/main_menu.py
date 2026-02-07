from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.concat_menu import ConcatMenu
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.real_time_menu import RealTimeMenu
from tts_audiobook_tool.options_menu import OptionsMenu
from tts_audiobook_tool.generate_menu import GenerateMenu
from tts_audiobook_tool.project_menu import ProjectMenu
from tts_audiobook_tool.tools_menu import ToolsMenu
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.text_menu import TextMenu
from tts_audiobook_tool.tts_model.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_menu.voice_menu_shared import VoiceMenuShared


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
            if state.prefs.project_dir:
                items.append(
                    MenuItem(
                        make_gen_label, on_generate, hotkey="g"
                    )
                )
            if state.prefs.project_dir:
                items.append(
                    MenuItem(make_concat_label, lambda _, __: ConcatMenu.menu(state), hotkey="c")
                )
            if state.prefs.project_dir:
                items.append(
                    MenuItem(
                        make_realtime_label, on_realtime, hotkey="r"
                    )
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
            path = state.project.qwen3_target or Qwen3BaseModel.DEFAULT_REPO_ID
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
    
    base_label = f"Voice clone and model settings"
    
    prefix, value = Tts.get_class().get_voice_display_info(state.project, Tts.get_instance_if_exists())
    
    if not value:
        combined = prefix
    else:
        combined = f"{prefix}: {value}"
    
    return f"{base_label} {COL_DIM}({combined}{COL_DIM})"

def on_voice(state: State, __) -> None:
    if not state.prefs.project_dir:
        return
    VoiceMenuShared.menu(state)

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
    return AppUtil.get_label_with_prereq_error(state.project, "Generate audio")

def on_generate(state: State, _: MenuItem) -> None:
    # Note, not blocking on other missing prereq types
    if not state.project.phrase_groups:
        print_feedback("Requires text", is_error=True)
    else:
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
    return AppUtil.get_label_with_prereq_error(state.project, "Realtime audio generation")

def on_realtime(state: State, _: MenuItem) -> None:
    if not state.project.phrase_groups:
        print_feedback("Requires text", is_error=True)
    else:
        RealTimeMenu.menu(state)

# Quit
def on_quit(_: State, __: MenuItem):
    print_feedback("State saved.", extra_line=False)
    exit(0)
