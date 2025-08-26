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

            s = f"Project directory {COL_DIM}(currently: {proj_dir or "none"})"
            print_heading(s)

            printt(f"{make_hotkey_string("1")} Create new project")
            printt(f"{make_hotkey_string("2")} Open existing project")
            if proj_dir:
                printt(f"{make_hotkey_string("3")} Open current project directory in UI {COL_DIM}({proj_dir})")
            printt()
            hotkey = ask_hotkey()
            if hotkey == "1":
                ProjectSubmenu.ask_and_set_new_project(state)
                continue
            elif hotkey == "2":
                ProjectSubmenu.ask_and_set_existing_project(state)
                continue
            elif hotkey == "3" and proj_dir:
                err = DirOpenUtil.open(proj_dir)
                if err:
                    ask_error(err)
                continue
            else:
                break



    @staticmethod
    def ask_and_set_new_project(state: State) -> None:
        """
        Asks user for directory and creates new project
        Returns True on success
        """
        s = "Enter empty directory path for new project:"
        s2 = "Select empty directory"
        dir_path = ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=False)
        if not dir_path:
            return
        err = state.make_new_project(dir_path)
        if err:
            ask_error(err)
            return

        hint_text = HINT_PROJECT_SUBDIRS.text
        hint_text = hint_text.replace("%1", os.path.join(dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR))
        hint_text = hint_text.replace("%2", os.path.join(dir_path, PROJECT_CONCAT_SUBDIR))
        hint = Hint(HINT_PROJECT_SUBDIRS.key, HINT_PROJECT_SUBDIRS.heading, hint_text)
        AppUtil.show_hint_if_necessary(state.prefs, hint, and_prompt=True)

        printt_cls("Project directory set.")
        return

    @staticmethod
    def ask_and_set_existing_project(state: State) -> None:
        """
        Asks user for directory and if valid, sets state to existing project
        Returns True on success
        """
        s = "Enter existing project directory path:"
        s2 = "Select existing project directory"
        dir = ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=True)
        if not dir:
            return
        err = Project.is_valid_project_dir(dir)
        if err:
            ask_error(err)
            return

        state.set_existing_project(dir)
        printt_cls("Project directory set.")
        return
