import os
from pathlib import Path
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.options_util import OptionsUtil
from tts_audiobook_tool.generate_validate_submenus import GenerateValidateSubmenus
from tts_audiobook_tool.project_util import ProjectUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.text_segments_util import TextSegmentsUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_util import VoiceUtil

class MainMenuUtil:
    """
    Main menu and misc submenus
    """

    @staticmethod
    def menu(state: State, did_reset=False) -> None:

        MainMenuUtil._print_menu(state, did_reset)

        Shared.mode = "menu"
        hotkey = ask_hotkey()
        Shared.mode = ""
        if Shared.stop_flag:
            Shared.stop_flag = False
            MainMenuUtil.quit()
        if not hotkey:
            return
        MainMenuUtil._handle_menu_hotkey(hotkey, state)


    @staticmethod
    def _print_menu(state: State, did_reset: bool):

        if MENU_CLEARS_SCREEN:
            os.system('cls' if os.name == 'nt' else 'clear')

        num_segments_complete = ProjectDirUtil.num_audio_segment_files(state)

        # Title
        model_name = "Oute TTS" if Shared.is_oute() else "Chatterbox TTS"
        print_heading(f"{APP_NAME} (active model: {model_name})")

        # Dir check
        if did_reset:
            printt(f"{COL_ERROR}Directory {state.prefs.project_dir} not found.\nCleared project settings.\n")

        num_audio_files = ProjectDirUtil.num_audio_segment_files(state)

        # Project
        s = f"{make_hotkey_string("P")} Project directory "
        if not state.prefs.project_dir:
            s += f"{COL_DIM}(Must set this first)"
        else:
            s += f"{COL_DIM}(current: {COL_ACCENT}{state.prefs.project_dir}{COL_DIM})"
        printt(s)

        # Voice
        if state.prefs.project_dir:
            pass
            name = "Oute" if Shared.is_oute() else "Chatterbox"
            s = f"{make_hotkey_string("V")} Voice clone ({name})"
            s += f"{COL_DIM}(current: {COL_ACCENT}{state.project.get_voice_label()}{COL_DIM})"
            printt(s)

        # Text
        if state.prefs.project_dir:
            s = f"{make_hotkey_string("T")} Text "
            if state.project.text_segments:
                s += f"{COL_DIM}(current: {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} segmented lines)"
            printt(s)

        # Generate audio
        if state.prefs.project_dir:
            s = f"{make_hotkey_string("G")} Generate audio"
            voice_name = "Oute" if Shared.is_oute() else "Chatterbox"
            if not state.project.has_voice and not state.project.text_segments:
                s2 = f"{COL_DIM} (must first set {voice_name} voice and text)"
            elif not state.project.has_voice:
                s2 = f"{COL_DIM} (must first set {voice_name} voice)"
            elif not state.project.text_segments:
                s2 = f"{COL_DIM} (must first set text)"
            else:
                s2 = f" {COL_DIM}({COL_ACCENT}{num_segments_complete}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} lines complete)"
            printt(s + s2)

        # Detect errors
        if state.prefs.project_dir and num_audio_files > 0:
            s = f"{make_hotkey_string("Y")} Validate generated audio"
            printt(s)

        # Concat
        if state.prefs.project_dir:
            s = f"{make_hotkey_string("C")} Combine audio segments and finalize"
            if num_audio_files == 0:
                s += f" {COL_DIM}(must first generate audio)"
            printt(s)

        # Options
        printt(f"{make_hotkey_string("O")} Options")

        # Quit
        printt(f"{make_hotkey_string("Q")} Quit")

        printt()


    @staticmethod
    def _handle_menu_hotkey(hotkey: str, state: State) -> None:

        num_audio_files = ProjectDirUtil.num_audio_segment_files(state)

        match hotkey:
            case "p":
                ProjectUtil.project_submenu(state)
            case "v":
                if state.prefs.project_dir:
                    VoiceUtil.voice_submenu(state)
            case "t":
                if not state.prefs.project_dir:
                    return
                if not state.project.text_segments:
                    TextSegmentsUtil.set_text_submenu(state)
                else:
                    MainMenuUtil.text_submenu(state)
            case "g":
                if state.project.can_generate_audio:
                    GenerateValidateSubmenus.generate_submenu(state)
            case "y":
                if not state.prefs.project_dir or num_audio_files == 0:
                    return
                GenerateValidateSubmenus.validate_submenu(state)
            case "c":
                if not state.prefs.project_dir or num_audio_files == 0:
                    return
                ConcatSubmenu.concat_submenu(state)
            case "o":
                OptionsUtil.options_submenu(state)
            case "q":
                MainMenuUtil.quit()


    @staticmethod
    def text_submenu(state: State) -> None:

        print_heading("Text:")
        printt(f"{make_hotkey_string("1")} View text")
        printt(f"{make_hotkey_string("2")} Replace text\n")
        hotkey = ask()
        if hotkey == "1":
            print_project_text(state)
            ask("Press enter: ")
        elif hotkey == "2":
            num_files = ProjectDirUtil.num_audio_segment_files(state)
            if num_files == 0:
                TextSegmentsUtil.set_text_submenu(state)
            else:
                s = f"Replacing text will invalidate {num_files} previously generated audio file fragments for this project.\nAre you sure? "
                if ask_hotkey(s):
                    TextSegmentsUtil.set_text_submenu(state)

    @staticmethod
    def quit():
        printt("State saved. Exiting")
        exit(0)

# ---

def print_project_text(state: State) -> None:

    index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
    indices = index_to_path.keys()
    texts = [item.text for item in state.project.text_segments]

    print_heading(f"Text segments ({COL_DEFAULT}{len(texts)}{COL_ACCENT}):")

    max_width = len(str(len(texts)))

    for i, text in enumerate(texts):
        s1 = make_hotkey_string( str(i+1).rjust(max_width) )
        s2 = "☑" if i in indices else "☐"
        printt(f"{s1}  {s2}    {text.strip()}")
    printt()

