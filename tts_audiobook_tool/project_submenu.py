from pathlib import Path
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State

class ProjectSubmenu:

    @staticmethod
    def submenu(state:State) -> None:
        """ Returns new | existing | empty string """

        print_heading("Project directory:")
        printt(f"{make_hotkey_string("1")} Create new project directory")
        printt(f"{make_hotkey_string("2")} Open existing project directory")
        printt()
        hotkey = ask_hotkey()
        if hotkey == "1":
            did = ProjectSubmenu.ask_and_set_new_project(state, state.project.oute_voice_json)
            if not did:
                ProjectSubmenu.submenu(state)
        elif hotkey == "2":
            did = ProjectSubmenu.ask_and_set_existing_project(state)
            if not did:
                ProjectSubmenu.submenu(state)


    @staticmethod
    def ask_and_set_new_project(state: State, previous_voice: dict | str | None) -> bool:
        """
        Asks user for directory and creates new project
        Returns True on success
        """
        s = "Enter empty directory path for new project:"
        s2 = "Select empty directory"
        path = ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=False)
        if not path:
            return False
        err = state.make_new_project(path)
        if err:
            ask_error(err)
            return False

        return True

        # TODO
        # if isinstance(previous_voice, dict):
        #     b = ask_confirm(f"Carry over voice data ({previous_voice.get("identifier", "voice")})? ")
        #     if b:
        #         VoiceUtil.save_to_project_dir_and_set_state(previous_voice, self.state)

    @staticmethod
    def ask_and_set_existing_project(state: State) -> bool:
        """
        Asks user for directory and if valid, sets state to existing project
        Returns True on success
        """
        s = "Enter existing project directory path:"
        s2 = "Select existing project directory"
        dir = ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=True)
        if not dir:
            return False
        err = Project.is_valid_project_dir(dir)
        if err:
            ask_error(err)
            return False

        dir = str( Path(dir).resolve() )
        state.set_existing_project(dir)
        return True
