import os
from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.options_submenu import OptionsSubmenu
from tts_audiobook_tool.generate_submenu import GenerateSubmenu
from tts_audiobook_tool.project_submenu import ProjectSubmenu
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.text_submenu import TextSubmenu
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_chatterbox_submenu import VoiceChatterboxSubmenu
from tts_audiobook_tool.voice_oute_submenu import VoiceOuteSubmenu

class MainMenu:
    """
    Main menu and misc submenus
    """

    @staticmethod
    def menu(state: State, did_reset=False) -> None:

        MainMenu._print_menu(state, did_reset)

        Shared.mode = "menu"
        hotkey = ask_hotkey()
        Shared.mode = ""
        if Shared.stop_flag:
            Shared.stop_flag = False
            MainMenu.quit()
        if not hotkey:
            return
        MainMenu._handle_menu_hotkey(hotkey, state)


    @staticmethod
    def _print_menu(state: State, did_reset: bool):

        if MENU_CLEARS_SCREEN:
            os.system('cls' if os.name == 'nt' else 'clear')

        # Title
        model_name = "Oute TTS" if Shared.is_oute() else "Chatterbox TTS"
        s = f"{COL_DIM}(active model: {COL_ACCENT}{model_name}{COL_DIM})"
        print_heading(f"{APP_NAME} {s}")

        # Dir check
        if did_reset:
            printt(f"{COL_ERROR}Directory {state.prefs.project_dir} not found.\nCleared project settings.\n")

        num_generated = state.project.sound_segments.num_generated()

        # Project
        s = f"{make_hotkey_string("P")} Project directory "
        if not state.prefs.project_dir:
            s += f"{COL_DIM}(Must set this first)"
        else:
            s += f"{COL_DIM}(currently: {COL_ACCENT}{state.prefs.project_dir}{COL_DIM})"
        printt(s)

        # Voice
        if state.prefs.project_dir:
            pass
            model_name = "Oute" if Shared.is_oute() else "Chatterbox"
            s = f"{make_hotkey_string("V")} Voice clone "
            s += f"{COL_DIM}(currently: {COL_ACCENT}{state.project.get_voice_label()}{COL_DIM})"
            printt(s)

        # Text
        if state.prefs.project_dir:
            s = f"{make_hotkey_string("T")} Text "
            if state.project.text_segments:
                s += f"{COL_DIM}({COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} lines)"
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
                s2 = f" {COL_DIM}({COL_ACCENT}{num_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} lines complete)"
            printt(s + s2)

        # Concat
        if state.prefs.project_dir:
            s = f"{make_hotkey_string("C")} Create audiobook file/s"
            if num_generated == 0:
                s += f" {COL_DIM}(must first generate audio)"
            printt(s)

        # Options
        printt(f"{make_hotkey_string("O")} Options, tools")

        # Quit
        printt(f"{make_hotkey_string("Q")} Quit")

        printt()


    @staticmethod
    def _handle_menu_hotkey(hotkey: str, state: State) -> None:

        num_generated = state.project.sound_segments.num_generated()

        match hotkey:
            case "p":
                ProjectSubmenu.project_submenu(state)
            case "v":
                if state.prefs.project_dir:
                    if Shared.is_oute():
                        VoiceOuteSubmenu.submenu(state)
                    elif Shared.is_chatterbox():
                        VoiceChatterboxSubmenu.submenu(state)
            case "t":
                if not state.prefs.project_dir:
                    return
                if not state.project.text_segments:
                    TextSubmenu.set_text_submenu(state, "Set text:")
                else:
                    TextSubmenu.submenu(state)
            case "g":
                if state.project.can_generate_audio:
                    GenerateSubmenu.generate_submenu(state)
            case "c":
                if not state.prefs.project_dir or num_generated == 0:
                    return
                ConcatSubmenu.submenu(state)
            case "o":
                OptionsSubmenu.submenu(state)
            case "q":
                MainMenu.quit()


    @staticmethod
    def quit():
        printt("State saved. Exiting.")
        exit(0)

