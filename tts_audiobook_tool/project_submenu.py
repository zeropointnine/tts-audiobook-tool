from pathlib import Path
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State

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
            ask_error(err)
            return

        if MENU_CLEARS_SCREEN:
            ask_continue("Project set.")

        # TODO
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
        err = Project.is_valid_project_dir(dir)
        if err:
            ask_error(err)
            return

        dir = str( Path(dir).resolve() )
        state.set_existing_project(dir)
