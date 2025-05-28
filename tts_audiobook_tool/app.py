import os
from pathlib import Path
import signal
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.generate_validate_submenus import GenerateValidateSubmenus
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.text_segments_util import TextSegmentsUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_util import VoiceUtil

class App:
    """
    Main app class.
    Runs a loop that prints menu, responds to menu selection.
    """

    def __init__(self):

        AppUtil.init_logging()
        signal.signal(signal.SIGINT, self.signal_handler)

        self.state = State()

    def signal_handler(self, signum, frame):
        def print_message(s: str):
            printt()
            printt(COL_ERROR + "*" * len(s))
            printt(s)
            printt(COL_ERROR + "*" * len(s))
            printt()

        match Shared.mode:
            case "generating":
                Shared.stop_flag = True
                print_message("Control-C pressed, will stop after current gen...")
            case "validating":
                Shared.stop_flag = True
                print_message("Control-C pressed, will stop")
            case "menu":
                Shared.stop_flag = True

    def loop(self):
        while True:

            # Dir check
            did_reset = False
            if self.state.prefs.project_dir and not os.path.exists(self.state.prefs.project_dir):
                did_reset = True
                self.state.reset()

            self.print_menu(did_reset=did_reset)

            Shared.mode = "menu"
            hotkey = ask_hotkey()
            Shared.mode = ""
            if Shared.stop_flag:
                Shared.stop_flag = False
                self.quit()
            if not hotkey:
                continue
            self.handle_hotkey(hotkey)

    def print_menu(self, did_reset=False):

        if MENU_CLEARS_SCREEN:
            os.system('cls' if os.name == 'nt' else 'clear')

        num_audio_segments_complete = ProjectDirUtil.num_audio_segment_files(self.state)

        # Title
        print_heading(f"Main menu - {APP_NAME}:")

        # Dir check
        if did_reset:
            printt(f"{COL_ERROR}Directory {self.state.prefs.project_dir} not found.\nCleared project settings.\n")

        num_audio_files = ProjectDirUtil.num_audio_segment_files(self.state)

        # Project
        s = f"{make_hotkey_string("P")} Project directory "
        if not self.state.prefs.project_dir:
            s += f"{COL_DIM}(Must set this first)"
        else:
            s += f"{COL_DIM}(current: {COL_ACCENT}{self.state.prefs.project_dir}{COL_DIM})"
        printt(s)

        # Voice
        if self.state.prefs.project_dir:
            s = f"{make_hotkey_string("V")} Voice "
            if self.state.project.voice:
                identifier = self.state.project.voice.get("identifier", "voice") if self.state.project.voice else "None"
                s += f"{COL_DIM}(current: {COL_ACCENT}{identifier}{COL_DIM}){Ansi.RESET}"
            printt(s)

        # Text
        if self.state.prefs.project_dir:
            s = f"{make_hotkey_string("T")} Text "
            if self.state.project.text_segments:
                s += f"{COL_DIM}(current: {COL_ACCENT}{len(self.state.project.text_segments)}{COL_DIM} lines)"
            printt(s)

        # Generate audio
        if self.state.prefs.project_dir:
            s = f"{make_hotkey_string("G")} Generate audio"
            if not self.state.project.voice and not self.state.project.text_segments:
                s2 = f"{COL_DIM} (must first set voice and text)"
            elif not self.state.project.voice:
                s2 = f"{COL_DIM} (must first set voice)"
            elif not self.state.project.text_segments:
                s2 = f"{COL_DIM} (must first set text)"
            else:
                s2 = f" {COL_DIM}({COL_ACCENT}{num_audio_segments_complete}{COL_DIM} of {COL_ACCENT}{len(self.state.project.text_segments)}{COL_DIM} lines complete)"
            printt(s + s2)

        # Detect errors
        if self.state.prefs.project_dir and num_audio_files > 0:
            s = f"{make_hotkey_string("Y")} Validate generated audio"
            printt(s)

        # Concat
        if self.state.prefs.project_dir:
            s = f"{make_hotkey_string("C")} Combine audio segments and finalize"
            if num_audio_files == 0:
                s += f" {COL_DIM}(must first generate audio)"
            printt(s)

        # Options
        printt(f"{make_hotkey_string("O")} Options")

        # Quit
        printt(f"{make_hotkey_string("Q")} Quit")

        printt()

    def handle_hotkey(self, hotkey: str) -> None:

        num_audio_files = ProjectDirUtil.num_audio_segment_files(self.state)

        match hotkey:
            case "p":
                project_type = self.project_dir_submenu()
                if project_type == "new":
                    self.ask_and_set_new_project(self.state.project.voice)
                elif project_type == "existing":
                    self.ask_and_set_existing_project()
            case "v":
                if self.can_set_voice:
                    VoiceUtil.voice_submenu(self.state, num_audio_files)
            case "t":
                if not self.state.prefs.project_dir:
                    return
                if not self.state.project.text_segments:
                    TextSegmentsUtil.set_text_submenu(self.state)
                else:
                    self.text_submenu()
            case "g":
                if self.can_generate_audio:
                    GenerateValidateSubmenus.generate_submenu(self.state)
            case "y":
                if not self.state.prefs.project_dir or num_audio_files == 0:
                    return
                GenerateValidateSubmenus.validate_submenu(self.state)
            case "c":
                if not self.state.prefs.project_dir or num_audio_files == 0:
                    return
                ConcatUtil.concat_submenu(self.state)
            case "o":
                self.options_submenu()
            case "q":
                self.quit()

    # ---

    @property
    def can_set_voice(self) -> bool:
        return bool(self.state.prefs.project_dir)

    @property
    def can_generate_audio(self) -> bool:
        return bool(self.state.project.voice and self.state.project.text_segments)

    # ---

    def project_dir_submenu(self) -> str:
        """ Returns new | existing | empty string """
        print_heading("Project directory:")
        printt(f"{make_hotkey_string("1")} Create new project directory")
        printt(f"{make_hotkey_string("2")} Open existing project directory")
        printt()
        hotkey = ask_hotkey()
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
                project_path = Path(inp)
                dir = str(project_path)
            except Exception as e:
                printt(f"Bad path: {e}", "error")
                return
            try:
                # Make project dir
                os.makedirs(project_path, exist_ok=True)
                # Make audio segment subdir
                audio_segments_path = project_path / AUDIO_SEGMENTS_SUBDIR
                os.makedirs(audio_segments_path, exist_ok=True)
                # Make concatenated subdir
                concat_path = project_path / CONCAT_SUBDIR
                os.makedirs(concat_path, exist_ok=True)
            except Exception as e:
                printt(f"Error creating directory: {e}", "error")
                return

        dir = str( Path(dir).resolve() )
        self.state.set_project(dir)

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
        self.state.set_project(dir)

    def text_submenu(self) -> None:
        print_heading("Text:")
        printt(f"{make_hotkey_string("1")} View text")
        printt(f"{make_hotkey_string("2")} Replace text\n")
        hotkey = ask()
        if hotkey == "1":
            AppUtil.print_project_text(self.state)
            ask("Press enter: ")
        elif hotkey == "2":
            num_files = ProjectDirUtil.num_audio_segment_files(self.state)
            if num_files == 0:
                TextSegmentsUtil.set_text_submenu(self.state)
            else:
                s = f"Replacing text will invalidate {num_files} previously generated audio file fragments for this project.\nAre you sure? "
                if ask_hotkey(s):
                    TextSegmentsUtil.set_text_submenu(self.state)

    def options_submenu(self) -> None:

        print_heading("Options:")
        printt(f"{make_hotkey_string("1")} Set temperature (currently: {self.state.prefs.temperature})")
        printt(f"{make_hotkey_string("2")} Normalize audio after generate (currently: {self.state.prefs.should_normalize})")
        printt(f"{make_hotkey_string("3")} Play audio after each segment is generated (currently: {self.state.prefs.play_on_generate})")
        printt(f"{make_hotkey_string("4")} Transcode FLAC to MP4, preserving app metadata")
        printt()

        hotkey = ask_hotkey()

        match hotkey:
            case "1":
                value = ask("Enter temperature (0.0 < value <= 2.0): ")
                try:
                    value = float(value)
                    if not (0.0 < value <= 2.0):
                        printt("Out of range", "error")
                    else:
                        self.state.prefs.temperature = value
                        ask("Changed. Press enter: ")
                        printt()
                except:
                    printt("Bad value", "error")
            case "2":
                self.state.prefs.should_normalize = not self.state.prefs.should_normalize
                printt(f"Normalize audio after generate set to: {self.state.prefs.should_normalize}")
                printt()
                if MENU_CLEARS_SCREEN:
                    ask_hotkey("Press enter: ")
            case "3":
                self.state.prefs.play_on_generate = not self.state.prefs.play_on_generate
                printt(f"Play audio after each segment is generated set to: {self.state.prefs.play_on_generate}")
                printt()
                if MENU_CLEARS_SCREEN:
                    ask_hotkey("Press enter: ")
            case "4":
                TranscodeUtil.ask_transcode(self.state)

    def quit(self):
        printt("State saved. Exiting")
        exit(0)
