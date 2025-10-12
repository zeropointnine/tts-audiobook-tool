from pathlib import Path
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.dir_open_util import DirOpenUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State

class ProjectSubmenu:

    @staticmethod
    def submenu(state:State) -> None:

        while True:

            proj_dir = state.project.dir_path

            s = make_currently_string(proj_dir or "none")
            print_heading(f"Project directory {s}")

            printt(f"{make_hotkey_string('1')} Start a new project")
            printt(f"{make_hotkey_string('2')} Open an existing project")
            if proj_dir:
                printt(f"{make_hotkey_string('3')} View current project directory in OS UI {COL_DIM}({proj_dir})")
            printt()
            hotkey = ask_hotkey()
            if hotkey == "1":
                did = ProjectSubmenu.ask_and_set_new_project(state)
                if did:
                    return
                else:
                    continue
            elif hotkey == "2":
                did = ProjectSubmenu.ask_and_set_existing_project(state)
                if did:
                    return
                else:
                    continue
            elif hotkey == "3" and proj_dir:
                err = DirOpenUtil.open(proj_dir)
                if err:
                    ask_error(err)
                continue
            else:
                break


    @staticmethod
    def ask_and_set_new_project(state: State) -> bool:
        """
        Asks user for directory and creates new project
        Returns True on success
        """
        s = "Enter the path to an empty directory:"
        s2 = "Select empty directory"
        dir_path = ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=False)
        if not dir_path:
            return False
        err = state.make_new_project(dir_path)
        if err:
            ask_error(err)
            return False

        AppUtil.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS, and_prompt=True)

        printt_set(f"Project directory set: {state.project.dir_path}")
        return True

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

        state.set_existing_project(dir)
        printt_set(f"Project directory set: {state.project.dir_path}")
        return True
