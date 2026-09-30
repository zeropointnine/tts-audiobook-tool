import os
from pathlib import Path

from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
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
            MenuItem("Make new project using settings from an existing tts-audiobook \"abr\" audiofile", on_make_new_project_using_abr),
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

        dir = ask.ask_dir_path(
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
            ask.ask_error(err)
            return False

        if migrate_current_settings and old_project.dir_path:
            ProjectTransferUtil.apply_project_settings(state.project, old_project)
            missing_paths = ProjectTransferUtil.copy_supporting_project_files(
                state.project,
                old_project.dir_path,
                ProjectTransferUtil.make_supporting_project_file_names(old_project)
            )
            state.project.save()
            state.set_existing_project(state.project.dir_path)
            ProjectNewMenu.print_missing_supporting_files_warning(missing_paths)

        hints.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS)

        print_feedback("Project directory set:", state.project.dir_path)
        ask.ask_enter_to_continue()

        return True

    @staticmethod
    def make_new_project_using_abr(state: State) -> bool:
        """
        Creates a new project using settings embedded in an existing ABR audio file.
        Returns True on success.
        Always ends with ask_util.ask_enter_to_continue(), except when the user
        cancels a path prompt, which ends with print_feedback("Cancelled") instead.
        """
        did_cancel = False
        dest_prepared = False
        dest_path = ""
        try:
            dest_dir = ask.ask_dir_path(
                console_message=(
                    "This will create a new project directory using settings from an\n"
                    "existing tts-audiobook-tool ABR audio file.\n\n"
                    "First, enter the path to an empty directory:"
                ),
                dialog_title="Select empty directory",
                initialdir=state.project.dir_path,
                mustexist=False
            )
            if not dest_dir:
                did_cancel = True
                return False

            abr_path = ask.ask_file_path(
                console_message="Now enter the path to the tts-audiobook-tool ABR audio file:",
                dialog_title="Select ABR audio file",
                filetypes=[('ABR audio files', '*.flac *.m4a *.m4b')],
                initialdir=state.project.dir_path,
            )
            if not abr_path:
                did_cancel = True
                return False

            if os.path.splitext(abr_path)[1].lower() not in {'.flac', '.m4a', '.m4b'}:
                ask.ask_error("Please select a .flac, .m4a, or .m4b file")
                return False

            app_meta = AppMetadata.load_from_file(abr_path)
            if app_meta is None:
                raw_meta = ProjectTransferUtil.load_raw_abr_metadata_string(abr_path)
                if not raw_meta:
                    printt(f"{COL_ERROR}This audio file has no ABR metadata")
                    printt()
                    return False

                parse_result = AppMetadata.get_from_json_string(raw_meta)
                detail = parse_result if isinstance(parse_result, str) else 'Could not read ABR metadata'
                ask.ask_error(f"Invalid ABR metadata: {detail}")
                return False

            project_snapshot = app_meta.project_snapshot
            if not project_snapshot:
                printt(f"{COL_ERROR}{ABR_OLD_VERSION_MESSAGE}")
                printt()
                return False

            try:
                snapshot_project = ProjectTransferUtil.validate_abr_snapshot(project_snapshot)
            except Exception as exc:
                ask.ask_error(f"Invalid ABR project snapshot: {make_error_string(exc)}")
                return False

            err = State.prepare_new_project_directory(dest_dir)
            if err:
                ask.ask_error(err)
                return False
            dest_path = str(Path(dest_dir).expanduser())
            dest_prepared = True

            # Build, save, and reload the new project before changing either
            # the selected project or the persisted preference. A failed import
            # must not silently replace the project the user was working on.
            new_project = Project(dir_path=dest_path)
            ProjectTransferUtil.apply_project_settings(new_project, snapshot_project)
            source_dir = ProjectTransferUtil.get_snapshot_source_dir(project_snapshot, abr_path)
            missing_paths = []
            if source_dir:
                missing_paths = ProjectTransferUtil.copy_supporting_project_files(
                    new_project,
                    source_dir,
                    ProjectTransferUtil.make_supporting_project_file_names(snapshot_project),
                    strict_copy_errors=True,
                )
            # With no source directory, never search the process's current
            # directory for coincidentally named project files.
            err = new_project.save()
            if err:
                raise ValueError(err)
            loaded_project = ProjectLoadUtil.load_using_dir_path(dest_path)
            if isinstance(loaded_project, str):
                raise ValueError(loaded_project)

            previous_dir = state.prefs.project_dir
            state.prefs.project_dir = dest_path
            err = state.prefs.save()
            if err:
                state.prefs.project_dir = previous_dir
                raise ValueError(err)
            # Keeps the ABR's originating model for the mismatch hint.
            try:
                state.project = loaded_project
            except Exception:
                state.prefs.project_dir = previous_dir
                state.prefs.save()
                raise

            print_feedback("Project directory set:", state.project.dir_path)

            if not source_dir:
                ProjectNewMenu.print_snapshot_source_dir_hint(project_snapshot)
            elif missing_paths:
                ProjectNewMenu.print_missing_supporting_files_warning(missing_paths)

            hints.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS)
            return True
        except Exception as e:
            detail = make_error_string(e)
            if dest_prepared:
                detail += f"\nThe previous project remains selected. Partial files may remain at {dest_path}."
            ask.ask_error(detail)
            return False
        finally:
            if did_cancel:
                print_feedback("\nCancelled")
            else:
                ask.ask_enter_to_continue()

    @staticmethod
    def print_snapshot_source_dir_hint(project_snapshot: dict) -> None:
        """
        Explains why supporting files could not be copied when the settings in
        an ABR file came from a project directory that does not exist here —
        typically because the file was made on another computer.
        """
        source_dir = ProjectTransferUtil.get_snapshot_source_dir_display(project_snapshot)
        if not source_dir:
            printt(f"{COL_DIM}No source project directory was found for this ABR file.")
        else:
            printt(f"{COL_DIM}The settings in this audio file came from a project at {source_dir},")
            printt("which is not a project directory on this computer.")
        printt("The settings were imported, but supporting files such as voice samples must be copied")
        printt(f"into the new project directory by hand.{COL_DEFAULT}")
        printt()

    @staticmethod
    def print_missing_supporting_files_warning(missing_paths: list[str]) -> None:

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