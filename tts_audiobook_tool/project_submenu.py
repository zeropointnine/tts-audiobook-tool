import os
from pathlib import Path
import signal
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.options_submenu import OptionsSubmenu
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.generate_validate_submenus import GenerateValidateSubmenus
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.text_submenu import TextSubmenu
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_chatterbox_submenu import VoiceChatterboxSubmenu

class ProjectSubmenu:

    @staticmethod
    def project_submenu(state:State) -> None:
        """ Returns new | existing | empty string """

        print_heading("Project directory:")
        printt(f"{make_hotkey_string("1")} Create new project directory")
        printt(f"{make_hotkey_string("2")} Open existing project directory")
        printt()
        hotkey = ask_hotkey()
        if hotkey == "1":
            ProjectSubmenu.ask_and_set_new_project(state, state.project.oute_voice_json)
        elif hotkey == "2":
            ProjectSubmenu.ask_and_set_existing_project(state)

    @staticmethod
    def ask_and_set_new_project(state: State, previous_voice: dict | str | None) -> None:
        """ Asks user for directory and creates new project if directory is ok, else prints error feedback """

        printt("Enter directory path for new project:")
        path = ask_path()
        if not path:
            return
        err = state.make_new_project(path)
        if err:
            printt(err, "error")
            return

        if MENU_CLEARS_SCREEN:
            ask_continue("Project set.")

        #x
        # if isinstance(previous_voice, dict):
        #     b = ask_confirm(f"Carry over voice data ({previous_voice.get("identifier", "voice")})? ")
        #     if b:
        #         VoiceUtil.save_to_project_dir_and_set_state(previous_voice, self.state)

    @staticmethod
    def ask_and_set_existing_project(state: State) -> None:
        """ Asks user for directory and if valid, sets state to existing project, else prints error feedback """

        printt("Enter directory path of existing project:")
        dir = ask_path()
        if not dir:
            return
        err = ProjectDirUtil.check_dir_valid(dir)
        if err:
            printt(err, "error")
            return

        dir = str( Path(dir).resolve() )
        state.set_existing_project(dir)
