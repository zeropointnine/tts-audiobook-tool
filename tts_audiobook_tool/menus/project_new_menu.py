import os

from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project_util import ProjectUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class ProjectNewMenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_make_new_project(_: State, item: MenuItem) -> bool:
            return ProjectNewMenu.make_new_project(state, migrate_current_settings=bool(item.data))

        def on_make_new_project_using_abr(_: State, __: MenuItem) -> None:
            ProjectNewMenu.make_new_project_using_abr(state)

        items = [
            MenuItem("Make new project", on_make_new_project, False),
            MenuItem("Make new project using current project's settings", on_make_new_project, True),
            MenuItem("Make new project using settings from existing tts-audiobook \"abr\" audiofile", on_make_new_project_using_abr),
        ]

        MenuUtil.menu(
            state,
            "New Project",
            items,
            breadcrumb="New Project",
        )

    @staticmethod
    def make_new_project(state: State, migrate_current_settings: bool = False) -> bool:
        """
        Asks user for directory and creates new project.
        Returns True on success.
        """
        console_message = "Enter the path to an empty directory:"
        if migrate_current_settings:
            console_message = (
                "This will create a new project directory, copying over the\n"
                " current project's settings including voice-clone data and text.\n\n"
                + console_message
            )
        ui_title = "Select empty directory"

        dir = AskUtil.ask_dir_path(
            console_message=console_message,
            dialog_title=ui_title,
            initialdir=state.project.dir_path,
            mustexist=False
        )

        if not dir:
            return False

        old_project = state.project

        err = state.make_and_set_new_project(dir)
        if err:
            AskUtil.ask_error(err)
            return False

        if migrate_current_settings and old_project.dir_path:
            ProjectUtil.apply_project_settings(state.project, old_project)
            missing_paths = ProjectUtil.copy_supporting_project_files(
                state.project,
                old_project.dir_path,
                ProjectUtil.make_supporting_project_file_names(old_project)
            )
            state.project.save()
            state.set_existing_project(state.project.dir_path)
            ProjectNewMenu.print_missing_supporting_files_warning(missing_paths)

        Hint.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS)

        print_feedback("Project directory set:", state.project.dir_path)
        AskUtil.ask_enter_to_continue()
        
        return True

    @staticmethod
    def make_new_project_using_abr(state: State) -> bool:
        """
        Creates a new project using settings embedded in an existing ABR audio file.
        Returns True on success.
        Always ends with AskUtil.ask_enter_to_continue().
        """
        try:
            dest_dir = AskUtil.ask_dir_path(
                console_message=(
                    "This will create a new project directory using settings from an\n"
                    "existing tts-audiobook-tool ABR audio file.\n\n"
                    "Enter the path to an empty directory:"
                ),
                dialog_title="Select empty directory",
                initialdir=state.project.dir_path,
                mustexist=False
            )
            if not dest_dir:
                return False

            abr_path = AskUtil.ask_file_path(
                console_message="Enter the path to the tts-audiobook-tool ABR audio file:",
                dialog_title="Select ABR audio file",
                filetypes=[('ABR audio files', '*.flac *.m4a *.m4b')],
                initialdir=state.project.dir_path,
            )
            if not abr_path:
                return False

            if os.path.splitext(abr_path)[1].lower() not in {'.flac', '.m4a', '.m4b'}:
                print_feedback("Please select a .flac, .m4a, or .m4b file", is_error=True)
                return False

            app_meta = AppMetadata.load_from_file(abr_path)
            if app_meta is None:
                raw_meta = ProjectUtil.load_raw_abr_metadata_string(abr_path)
                if not raw_meta:
                    printt(f"{COL_ERROR}This audio file has no ABR metadata")
                    printt()
                    return False

                printt(f"{COL_ERROR}{ABR_OLD_VERSION_MESSAGE}")
                printt()
                return False

            project_snapshot = app_meta.project_snapshot
            if not project_snapshot:
                printt(f"{COL_ERROR}{ABR_OLD_VERSION_MESSAGE}")
                printt()
                return False

            err = state.make_and_set_new_project(dest_dir)
            if err:
                print_feedback(err, is_error=True)
                return False

            snapshot_project = ProjectUtil.make_project_from_snapshot(
                state.project.dir_path,
                project_snapshot
            )
            ProjectUtil.apply_project_settings(state.project, snapshot_project)

            missing_paths = ProjectUtil.copy_supporting_project_files(
                state.project,
                ProjectUtil.get_snapshot_source_dir(project_snapshot),
                ProjectUtil.make_supporting_project_file_names(snapshot_project)
            )

            state.project.save()
            state.set_existing_project(state.project.dir_path)

            print_feedback("Project directory set:", state.project.dir_path)

            if missing_paths:
                ProjectNewMenu.print_missing_supporting_files_warning(missing_paths)

            Hint.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS)
            return True
        except Exception as e:
            print_feedback(make_error_string(e), is_error=True)
            return False
        finally:
            AskUtil.ask_enter_to_continue()

    @staticmethod
    def print_missing_supporting_files_warning(missing_paths: list[str]) -> None:
        
        # Special case: Oute, not worth reasoning through this
        if len(missing_paths) == 1 and "default.json" in missing_paths[0]:
            missing_paths = []

        if not missing_paths:
            return
        
        printt(
            f"{COL_ERROR}Note that following supporting project files do not exist and were not copied over:{COL_DEFAULT}"
        )
        for path in missing_paths:
            printt(f"- {path}")
        printt()


ABR_OLD_VERSION_MESSAGE = (
    "Audio file was generated with an old version of tts-audiobook-tool (pre-5-2026)\n"
    "which does not contain project snapshot data."
)