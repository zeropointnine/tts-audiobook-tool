import os
from pathlib import Path
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.bad_gen_util import BadGenUtil
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

        # Project
        s = f"{make_hotkey_string("P")} Project directory "
        if not self.state.project_dir:
            s += f"{COL_DIM}(Must set this first)"
        else:
            s += f"{COL_DIM}(current: {COL_DEFAULT}{self.state.project_dir}{COL_DIM})"
        printt(s)

        # Voice
        if self.state.project_dir:
            s = f"{make_hotkey_string("V")} Voice "
            if self.state.voice:
                identifier = self.state.voice.get("identifier", "voice") if self.state.voice else "None"
                s += f"{COL_DIM}(current: {COL_DEFAULT}{identifier}{COL_DIM}){Ansi.RESET}"
            printt(s)

        # Text
        if self.state.project_dir:
            s = f"{make_hotkey_string("T")} Text "
            if self.state.text_segments:
                s += f"{COL_DIM}(current: {COL_DEFAULT}{len(self.state.text_segments)}{COL_DIM} lines)"
            printt(s)

        # View text
        if self.state.project_dir:
            s = f"{make_hotkey_string("U")} View text "
            if not self.state.text_segments:
                s += f"{COL_DIM}(must first set text)"
            printt(s)

        # Generate audio
        if self.state.project_dir:
            s = f"{make_hotkey_string("A")} Generate audio"
            if not self.state.voice and not self.state.text_segments:
                s2 = f"{COL_DIM} (must first set voice and text)"
            elif not self.state.voice:
                s2 = f"{COL_DIM} (must first set voice)"
            elif not self.state.text_segments:
                s2 = f"{COL_DIM} (must first set text)"
            else:
                num_complete = ProjectDirUtil.num_audio_segment_files(self.state)
                s2 = f" {COL_DIM}({COL_DEFAULT}{num_complete}{COL_DIM} of {COL_DEFAULT}{len(self.state.text_segments)}{COL_DIM} complete)"
            printt(s + s2)

        # Options
        s = f"{make_hotkey_string("Z")} Options, Utils"
        printt(s)

        printt()

    def handle_hotkey(self, hotkey: str) -> None:
        match hotkey:
            case "p":
                project_type = self.ask_project_type()
                if project_type == "new":
                    self.ask_and_set_new_project(self.state.voice)
                elif project_type == "existing":
                    self.ask_and_set_existing_project()
            case "v":
                if self.can_set_voice:
                    count = ProjectDirUtil.num_audio_segment_files(self.state)
                    if count:
                        s = f"Replacing voice will invalidate {count} previously generated audio file fragments for this project.\nAre you sure? "
                        if ask_hotkey(s):
                            VoiceUtil.ask_voice_and_set(self.state)
                    else:
                        VoiceUtil.ask_voice_and_set(self.state)
            case "t":
                if self.can_set_text:
                    count = ProjectDirUtil.num_audio_segment_files(self.state)
                    if count:
                        s = f"Replacing text will invalidate {count} previously generated audio file fragments for this project.\nAre you sure? "
                        if ask_hotkey(s):
                            TextSegmentsUtil.ask_text_segments_and_set(self.state)
                    else:
                        TextSegmentsUtil.ask_text_segments_and_set(self.state)
            case "u":
                if self.can_view_text:
                    print_text_segments(self.state.text_segments)
                    ask("Press enter: ")
                    printt()
            case "a":
                if self.can_generate_audio:
                    abort = GenerateUtil.ask_generate_all(self.state)
                    if abort:
                        return
                    ConcatUtil.concatenate_project_flacs(self.state)
                    ask("Press enter: ")
            case "z":
                self.print_ask_options()

    # ---

    @property
    def can_set_voice(self) -> bool:
        return bool(self.state.project_dir)

    @property
    def can_set_text(self) -> bool:
        return bool(self.state.project_dir)

    @property
    def can_view_text(self) -> bool:
        return bool(self.state.text_segments)

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

    def print_ask_options(self):

        printt(f"{COL_ACCENT}Options, utilities:\n")
        printt(f"{make_hotkey_string("1")} Temperature (currently: {self.state.temperature})")
        printt(f"{make_hotkey_string("2")} Play audio after each segment is generated (currently: {self.state.play_on_generate})")
        s = f"{make_hotkey_string("3")} Util: Detect audio generation errors "
        num_audio_segments = ProjectDirUtil.num_audio_segment_files(self.state)
        if num_audio_segments == 0:
            s += f"{COL_DIM}(Must first generate audio)"
        printt(s)
        printt(f"{make_hotkey_string("4")} Util: Concatenate FLAC files in a directory")

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
                if num_audio_segments > 0:
                    should_regen = BadGenUtil.ask_detect(self.state)
                    if should_regen:
                        GenerateUtil.ask_generate_all(self.state, skip_ask=True)
                        ask("Press enter: ")
            case "4":
                ConcatUtil.ask_do_dir_concat()
