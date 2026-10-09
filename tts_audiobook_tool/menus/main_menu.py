from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.menus.chat_menu import ChatMenu
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.real_time_playback_menu import RealTimePlaybackMenu
from tts_audiobook_tool.menus.options_menu import OptionsMenu
from tts_audiobook_tool.menus.generate_menu import GenerateMenu
from tts_audiobook_tool.menus.project_menu import ProjectMenu
from tts_audiobook_tool.menus.tools_menu import ToolsMenu
from tts_audiobook_tool.app_support import app_hint_util, hints
from tts_audiobook_tool.constants_hints import (
    HINT_CHATTERBOX_MULTILINGUAL_V3,
    HINT_LLM_API_KEY_REMOVED,
    HINT_SGL_OMNI_URL,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.menus.text_menu import TextMenu
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared


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
                ask.ask_error(f"Project directory {state.prefs.project_dir} not found.")
                state.reset()

            MainMenu.menu(state)

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    "Project", lambda _, __: ProjectMenu.menu(state), hotkey="p",
                    superlabel="Main", superlabel_no_blank_line=True
                )
            )
            items.append(
                MenuItem(make_text_label, on_text, hotkey="t")
            )
            items.append(
                MenuItem(make_voice_label, on_voice, hotkey="v")
            )
            items.append(
                MenuItem(make_model_label, on_model, hotkey="m")
            )
            items.append(
                MenuItem(
                    "Generate sound segments", on_generate, hotkey="g"
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

        def on_shown() -> None:
            # These hints may only appear the first time the main menu is shown
            is_first_show = not state.has_shown_main_menu
            state.mark_main_menu_shown()
            if is_first_show:
                # What's changed since the last run (shown before any other hints)
                app_hint_util.show_startup_version_messages(state.prefs)

                # One-time informational startup hints (tkinter, long paths, etc)
                app_hint_util.show_shared_startup_hints(state.prefs, is_server=False)

                if state.prefs.legacy_llm_api_key_removed:
                    hints.show_hint_if_necessary(state.prefs, HINT_LLM_API_KEY_REMOVED)

                if (
                    state.project.get_tts_model_type().id == "chatterbox_local"
                    and state.project.get_model_setting('chatterbox_local', 'type') == ChatterboxType.MULTILINGUAL_V2
                ):
                    hints.show_hint_if_necessary(
                        state.prefs, HINT_CHATTERBOX_MULTILINGUAL_V3
                    )

                if (
                    Tts.is_remote_mode()
                    and not state.prefs.remote_tts_url
                ):
                    hints.show_hint_if_necessary(state.prefs, HINT_SGL_OMNI_URL)

        heading = text_util.make_terminal_hyperlink(APP_URL, APP_NAME)
        MenuUtil.menu(
            state,
            heading,
            make_items,
            is_submenu=False,
            one_shot=True,
            breadcrumb="Main",
            on_shown=on_shown,
        )

# ---

def get_heading_tts_text(state: State) -> str:

    s = Tts.get_model_support(state.project).get_menu_text(state.project, None)
    if not Tts.is_remote_mode():
        return s
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
    snapshot = RemoteTtsDiscovery.get_snapshot()
    backend = ("audio.cpp" if snapshot.backend_kind == TtsBackendKind.AUDIO_CPP else
               "SGL-Omni" if snapshot.backend_kind == TtsBackendKind.SGL_OMNI else "Remote TTS")
    if Tts.get_active_type() == state.project.get_tts_model_type() and Tts._selected_server_model_id:
        s += f" {COL_DIM}{backend} server model id: {Tts._selected_server_model_id}{COL_ACCENT}"
    else:
        s += f" {COL_ERROR}{backend}: {Tts._remote_issue or (snapshot.issue.message if snapshot.issue else 'select a server model')}{COL_ACCENT}"
    return s

# ---

# Voice
def make_voice_label(state: State) -> str:
    return "Voice clone"

def on_voice(state: State, __) -> None:
    Tts.bind_project(state.project)
    if state.project.get_tts_model_type().id == "none":
        ask.ask_error("Requires TTS model")
        return
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    VoiceMenuShared.menu(state)

# Model settings
def make_model_label(state: State) -> str:
    suffix = ""
    if (
        state.project.dir_path
        and state.project.get_tts_model_type().id == "none"
        and len(Tts.get_available_tts_models()) >= 2
    ):
        suffix = f" {COL_ERROR}(requires: TTS model selection)"
    return f"Model settings{suffix}"

def on_model(state: State, __: MenuItem) -> None:
    Tts.bind_project(state.project)
    # In remote mode, the model picker lives inside Model settings,
    # so an unselected model must not block entry.
    if state.project.get_tts_model_type().id == "none" and not Tts.is_remote_mode():
        ask.ask_error(REQUIRES_TTS_MODEL)
        return
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    ModelMenuShared.menu(state)

# Text
def make_text_label(state: State) -> str:
    return "Text"

def on_text(state: State, __) -> None:
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    TextMenu.menu(state)

def on_generate(state: State, _: MenuItem) -> None:
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    GenerateMenu.menu(state)

def on_concat(state: State, _: MenuItem) -> None:
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    ConcatMenu.menu(state)

def on_realtime_audiobook(state: State, _: MenuItem) -> None:
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    RealTimePlaybackMenu.menu(state)

def on_chat(state: State, _: MenuItem) -> None:
    Tts.bind_project(state.project)
    if state.project.get_tts_model_type().id == "none":
        ask.ask_error(REQUIRES_TTS_MODEL)
        return
    if not state.project.dir_path:
        ask.ask_error(REQUIRES_PROJECT)
        return
    ChatMenu.menu(state)

# Quit
def on_quit(_: State, __: MenuItem):
    print_feedback("State saved.", extra_line=False, skip_pause=True)
    exit(0)

REQUIRES_PROJECT = "Requires a project"
REQUIRES_TTS_MODEL = "Requires TTS model"
