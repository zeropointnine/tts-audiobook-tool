from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.dir_open_util import DirOpenUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State

class ProjectSubmenu:

    @staticmethod
    def menu(state:State) -> None:

        # 1
        def on_new(_, __) -> bool:
            did = ProjectSubmenu.ask_and_set_new_project(state)
            if did:
                print_feedback(f"Project directory set: {state.project.dir_path}")
                return True
            return False

        # 2
        def on_existing(_, __) -> bool:
            did = ProjectSubmenu.ask_and_set_existing_project(state)
            if did:
                print_feedback(f"Project directory set: {state.project.dir_path}")
                AppUtil.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS, and_prompt=True)
                return True
            return False

        # 3
        def make_view_label(_) -> str:
            return f"View current project directory in OS UI {COL_DIM}({state.project.dir_path})"

        def on_view(_, __) -> None:
            err = DirOpenUtil.open(state.project.dir_path)
            if err:
                AskUtil.ask_error(err)

        # Menu
        def make_heading(_) -> str:
            s = make_currently_string(state.project.dir_path or "none")
            return f"Project directory {s}"

        def items_maker(_) -> list[MenuItem]:
            items = [
                MenuItem("New project", on_new),
                MenuItem("Open existing project", on_existing)
            ]
            if state.project.dir_path:
                items.append(
                    MenuItem(make_view_label, on_view)
                )
            return items

        MenuUtil.menu(state, make_heading, items_maker)


    @staticmethod
    def ask_and_set_new_project(state: State) -> bool:
        """
        Asks user for directory and creates new project
        Returns True on success
        """
        s = "Enter the path to an empty directory:"
        s2 = "Select empty directory"
        dir_path = AskUtil.ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=False)
        if not dir_path:
            return False
        err = state.make_new_project(dir_path)
        if err:
            AskUtil.ask_error(err)
            return False
        return True

    @staticmethod
    def ask_and_set_existing_project(state: State) -> bool:
        """
        Asks user for directory and if valid, sets state to existing project
        Returns True on success
        """
        s = "Enter existing project directory path:"
        s2 = "Select existing project directory"
        dir = AskUtil.ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=True)
        if not dir:
            return False
        err = Project.is_valid_project_dir(dir)
        if err:
            AskUtil.ask_error(err)
            return False

        state.set_existing_project(dir)
        return True
