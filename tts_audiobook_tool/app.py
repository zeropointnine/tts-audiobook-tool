import os
from pathlib import Path
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.verify_util import VerifyUtil
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.text_segments_util import TextSegmentsUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_util import VoiceUtil

class App:
    """
    Main app class.
    Runs a loop that prints menu, responds to menu selection, and updates program state.
    """

    def __init__(self):
        AppUtil.init_app_logging()
        self.state = State()

    def loop(self):
        while True:

            # Dir check
            did_reset = False
            if self.state.project_dir and not os.path.exists(self.state.project_dir):
                did_reset = True
                self.state.reset()
                self.state.save_project_dir_setting()

            self.print_menu(did_reset=did_reset)

            hotkey = ask_hotkey()
            if not hotkey:
                continue
            self.handle_hotkey(hotkey)

    def print_menu(self, did_reset=False):

        if MENU_CLEAR_SCREEN:
            os.system('cls' if os.name == 'nt' else 'clear')

        # Title
        printt(f"{COL_ACCENT}Menu - {APP_NAME}\n")

        # Dir check
        if did_reset:
            printt(f"{COL_ERROR}Directory {self.state.project_dir} not found.\nCleared project settings.\n")

        num_audio_files = ProjectDirUtil.num_audio_segment_files(self.state)

        # Project
        s = f"{make_hotkey_string("P")} Project directory "
        if not self.state.project_dir:
            s += f"{COL_DIM}(Must set this first)"
        else:
            s += f"{COL_DIM}(current: {COL_ACCENT}{self.state.project_dir}{COL_DIM})"
        printt(s)

        # Voice
        if self.state.project_dir:
            s = f"{make_hotkey_string("V")} Voice "
            if self.state.voice:
                identifier = self.state.voice.get("identifier", "voice") if self.state.voice else "None"
                s += f"{COL_DIM}(current: {COL_ACCENT}{identifier}{COL_DIM}){Ansi.RESET}"
            printt(s)

        # Text
        if self.state.project_dir:
            s = f"{make_hotkey_string("T")} Text "
            if self.state.text_segments:
                s += f"{COL_DIM}(current: {COL_ACCENT}{len(self.state.text_segments)}{COL_DIM} lines)"
            printt(s)

        # Generate audio
        if self.state.project_dir:
            s = f"{make_hotkey_string("G")} Generate audio"
            if not self.state.voice and not self.state.text_segments:
                s2 = f"{COL_DIM} (must first set voice and text)"
            elif not self.state.voice:
                s2 = f"{COL_DIM} (must first set voice)"
            elif not self.state.text_segments:
                s2 = f"{COL_DIM} (must first set text)"
            else:
                num_complete = ProjectDirUtil.num_audio_segment_files(self.state)
                s2 = f" {COL_DIM}({COL_ACCENT}{num_complete}{COL_DIM} of {COL_ACCENT}{len(self.state.text_segments)}{COL_DIM} lines complete)"
            printt(s + s2)

        # Detect errors
        if self.state.project_dir and num_audio_files > 0:
            s = s = f"{make_hotkey_string("Y")} Verify generated audio"
            printt(s)

        # Options
        s = f"{make_hotkey_string("Z")} Options, Utils"
        printt(s)

        printt()

    def handle_hotkey(self, hotkey: str) -> None:

        num_audio_files = ProjectDirUtil.num_audio_segment_files(self.state)

        match hotkey:
            case "p":
                project_type = self.ask_project_type()
                if project_type == "new":
                    self.ask_and_set_new_project(self.state.voice)
                elif project_type == "existing":
                    self.ask_and_set_existing_project()
            case "v":
                if self.can_set_voice:
                    if num_audio_files:
                        s = f"Replacing voice will invalidate {num_audio_files} previously generated audio file fragments for this project.\nAre you sure? "
                        if ask_hotkey(s):
                            VoiceUtil.ask_voice_and_set(self.state)
                    else:
                        VoiceUtil.ask_voice_and_set(self.state)
            case "t":
                if not self.state.project_dir:
                    return
                if not self.state.text_segments:
                    TextSegmentsUtil.ask_text_segments_and_set(self.state)
                else:
                    self.text_submenu()
            case "g":
                if self.can_generate_audio:
                    GenerateUtil.go(self.state, [], should_ask=True)
            case "y":
                if not self.state.project_dir or num_audio_files == 0:
                    return
                VerifyUtil.verify(self.state)
            case "z":
                self.options_submenu()

    # ---

    @property
    def can_set_voice(self) -> bool:
        return bool(self.state.project_dir)

    @property
    def can_generate_audio(self) -> bool:
        return bool(self.state.voice and self.state.text_segments)

    # ---

    def ask_project_type(self) -> str:
        """ Returns new | existing | empty string """
        printt(f"{make_hotkey_string("1")} Create new project directory")
        printt(f"{make_hotkey_string("2")} Open existing project directory")
        hotkey = ask_hotkey()
        printt()
        if hotkey == "1":
            return "new"
        elif hotkey == "2":
            return "existing"
        else:
            return ""

    def ask_and_set_new_project(self, previous_voice: dict | None) -> None:
        """ Asks user for directory and creates new project if directory is ok, else prints error feedback """

        printt("Enter directory path for new project:")
        inp = ask()
        if not inp:
            return

        if os.path.exists(inp):
            if os.listdir(inp):
                printt("Directory is not empty", "error")
                return
            else:
                dir = inp
        else:
            try:
                path = Path(inp)
                dir = str(path)
            except Exception as e:
                printt(f"Bad path: {e}", "error")
                return
            try:
                os.makedirs(path)
            except Exception as e:
                printt(f"Error creating directory: {e}", "error")
                return

        dir = str( Path(dir).resolve() )
        self.state.set_new_project_dir(dir)

        if isinstance(previous_voice, dict):
            b = ask_confirm(f"Carry over voice data ({previous_voice.get("identifier", "voice")})? ")
            if b:
                VoiceUtil.save_to_project_dir_and_set_state(previous_voice, self.state)

    def ask_and_set_existing_project(self) -> None:
        """ Asks user for directory and if valid, sets state to existing project, else prints error feedback """

        printt("Enter directory path of existing project:")
        dir = ask()
        if not dir:
            return
        err = ProjectDirUtil.check_project_dir_valid(dir)
        if err:
            printt(err, "error")
            return

        dir = str( Path(dir).resolve() )
        err = self.state.set_existing_project_dir(dir)
        if err:
            printt(err, "error")
            return

    def text_submenu(self) -> None:
        printt(f"{COL_ACCENT}Text:\n")
        printt(f"{make_hotkey_string("1")} View text")
        printt(f"{make_hotkey_string("2")} Replace text\n")
        hotkey = ask()
        if hotkey == "1":
            AppUtil.print_text_segments(self.state.text_segments)
            ask("Press enter: ")
        elif hotkey == "2":
            num_files = ProjectDirUtil.num_audio_segment_files(self.state)
            if num_files == 0:
                TextSegmentsUtil.ask_text_segments_and_set(self.state)
            else:
                s = f"Replacing text will invalidate {num_files} previously generated audio file fragments for this project.\nAre you sure? "
                if ask_hotkey(s):
                    TextSegmentsUtil.ask_text_segments_and_set(self.state)


    def options_submenu(self) -> None:

        printt(f"{COL_ACCENT}Options, utilities:\n")
        printt(f"{make_hotkey_string("1")} Temperature (currently: {self.state.temperature})")
        printt(f"{make_hotkey_string("2")} Play audio after each segment is generated (currently: {self.state.play_on_generate})")
        printt(f"{make_hotkey_string("3")} Util: Concatenate FLAC files")

        printt(f"{make_hotkey_string("Q")} Quit")
        printt()

        hotkey = ask_hotkey()
        match hotkey:
            case "q":
                exit(0)
            case "1":
                value = ask("Enter temperature (0.0 < value <= 2.0): ")
                try:
                    value = float(value)
                    if not (0.0 < value <= 2.0):
                        printt("Out of range", "error")
                    else:
                        self.state.temperature = value
                        self.state.save_pref_settings()
                        ask("Changed. Press enter: ")
                        printt()
                except:
                    printt("Bad value", "error")
            case "2":
                self.state.play_on_generate = not self.state.play_on_generate
                self.state.save_pref_settings()
                printt(f"Play audio after each segment is generated set to: {self.state.play_on_generate}")
                if MENU_CLEAR_SCREEN:
                    ask_hotkey("Press enter: ")
                printt()
            case "3":
                ConcatUtil.ask_do_dir_concat()
