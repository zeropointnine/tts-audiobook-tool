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
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil, SupportingFilesCopyResult
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class ProjectNewMenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_make_new_project(_: State, item: MenuItem) -> bool:
            return ProjectNewMenu.make_new_project(state, migrate_current_settings=bool(item.data))

        def on_make_new_project_using_abr(_: State, __: MenuItem) -> None:
            ProjectNewMenu.make_new_project_using_abr(state)

        items = [MenuItem("Make new project", on_make_new_project, False)]
        # Only offer settings migration when a project is currently loaded
        if state.project.dir_path:
            items.append(
                MenuItem("Make new project using current project's settings", on_make_new_project, True)
            )
        items.append(
            MenuItem("Make new project using settings from an existing tts-audiobook \"abr\" audiofile", on_make_new_project_using_abr)
        )

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
                " current project's settings including voice-clone data, text,\n"
                " and speaker assignment data.\n\n"
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
            text_file_names, voice_file_names = ProjectTransferUtil.collect_supporting_project_file_names(old_project)
            optional_file_names = set(ProjectTransferUtil.OPTIONAL_SUPPORTING_FILE_NAMES)
            if not old_project.phrase_groups:
                # A project with no text yet has no text file to miss.
                optional_file_names.add(PROJECT_TEXT_FILE_NAME)
            transfer = ProjectTransferUtil.transfer_supporting_project_files(
                state.project,
                old_project.dir_path,
                text_file_names,
                voice_file_names,
                optional_file_names=optional_file_names,
            )
            speaker_assign_errors = SpeakerAssignFileUtil.copy_files(old_project.dir_path, state.project.dir_path)
            state.project.save()
            state.set_existing_project(state.project.dir_path)
            ProjectNewMenu.print_missing_supporting_files_warning(transfer.missing_paths)
            ProjectNewMenu.print_supporting_files_todo(
                state.project.dir_path, transfer.missing_voice_dest_paths, bool(transfer.missing_text_paths),
            )
            for error in speaker_assign_errors:
                printt(f"{COL_ERROR}{error}{COL_DEFAULT}")
            if speaker_assign_errors:
                printt()

        hints.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS)

        print_feedback("Project directory set:", state.project.dir_path)
        ask.ask_enter_to_continue()

        return True

    @staticmethod
    def make_new_project_using_abr(state: State) -> bool:
        """
        Creates a new project using settings embedded in an existing ABR audio file.
        Returns True on success.
        Ends with ask_enter_to_continue() on success and ask_error() on failure
        (which itself waits for enter), except when the user cancels a path
        prompt, which ends with print_feedback("Cancelled") instead.
        """
        did_cancel = False
        did_succeed = False
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
                    ask.ask_error("This audio file has no ABR metadata")
                    return False

                parse_result = AppMetadata.get_from_json_string(raw_meta)
                detail = parse_result if isinstance(parse_result, str) else 'Could not read ABR metadata'
                ask.ask_error(f"Invalid ABR metadata: {detail}")
                return False

            project_snapshot = app_meta.project_snapshot
            if not project_snapshot:
                ask.ask_error(ABR_OLD_VERSION_MESSAGE)
                return False

            try:
                snapshot_project = ProjectTransferUtil.validate_abr_snapshot(project_snapshot, prompt_on_migration=True)
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
            text_file_names, voice_file_names = ProjectTransferUtil.collect_supporting_project_file_names(snapshot_project)
            transfer: SupportingFilesCopyResult | None = None
            if source_dir:
                transfer = ProjectTransferUtil.transfer_supporting_project_files(
                    new_project,
                    source_dir,
                    text_file_names,
                    voice_file_names,
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

            # The ABR holds settings only, so the text comes solely from a
            # copied project_text.json; judge by what actually loaded.
            needs_text = not loaded_project.phrase_groups
            if transfer is None:
                voice_dest_paths = [
                    ProjectTransferUtil.get_voice_file_dest_relative_path(name) for name in voice_file_names
                ]
                ProjectNewMenu.print_snapshot_source_dir_hint(
                    project_snapshot, dest_path, voice_dest_paths, needs_text,
                )
            else:
                # Name the source even on full success: it may have been
                # inferred from the ABR file's location rather than recorded.
                ProjectNewMenu.print_copied_supporting_files(source_dir, transfer.copied)
                if transfer.missing_paths:
                    ProjectNewMenu.print_missing_supporting_files_warning(transfer.missing_paths)
                ProjectNewMenu.print_supporting_files_todo(
                    dest_path, transfer.missing_voice_dest_paths, needs_text,
                )

            hints.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS)
            did_succeed = True
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
            elif did_succeed:
                ask.ask_enter_to_continue()

    @staticmethod
    def print_snapshot_source_dir_hint(
        project_snapshot: dict,
        dest_dir: str,
        voice_dest_paths: list[str],
        needs_text: bool,
    ) -> None:
        """
        Explains that nothing besides settings was imported when the source
        project of an ABR file could not be found — typically because the file
        was made on another computer — and what remains to be done by hand.
        Prints nothing when the new project needs nothing more.
        """
        if not voice_dest_paths and not needs_text:
            return

        missing_things = []
        if voice_dest_paths:
            missing_things.append("voice samples")
        if needs_text:
            missing_things.append("book text")
        verb = "were" if voice_dest_paths else "was"
        printt(f"{COL_ERROR}The project's {' and '.join(missing_things)} {verb} not copied.{COL_DEFAULT}")

        source_dir = ProjectTransferUtil.get_snapshot_source_dir_display(project_snapshot)
        if source_dir:
            printt(f"The settings came from a project at {source_dir},")
            printt("which is not available on this computer, and no project directory")
            printt("was found next to the audio file.")
        else:
            printt("The audio file does not record which project its settings came from,")
            printt("and no project directory was found next to it.")
        printt(f"{COL_DIM}(An ABR audio file stores project settings only, not voice samples or text.){COL_DEFAULT}")
        printt()
        ProjectNewMenu.print_supporting_files_todo(dest_dir, voice_dest_paths, needs_text)

    @staticmethod
    def print_supporting_files_todo(dest_dir: str, voice_dest_paths: list[str], needs_text: bool) -> None:
        """
        Tells the user what to put in place by hand to finish a new project
        whose supporting files could not all be copied.

        `voice_dest_paths` are project-relative paths under the voice subdir.
        """
        if not voice_dest_paths and not needs_text:
            return

        printt("To finish setting up the new project:")
        if voice_dest_paths:
            voice_dir = os.path.join(dest_dir, PROJECT_VOICE_SUBDIR)
            printt(f"- Copy these voice files into {COL_ACCENT}{voice_dir}{os.sep}{COL_DEFAULT}")
            prefix = PROJECT_VOICE_SUBDIR + "/"
            for path in voice_dest_paths:
                printt(f"    {path.removeprefix(prefix)}")
        if needs_text:
            printt("- Import the book text from the Text menu")
        printt()

    @staticmethod
    def print_copied_supporting_files(source_dir: str, copied: list[str]) -> None:
        if not copied:
            return
        printt(f"Copied supporting files from {COL_ACCENT}{source_dir}{COL_DEFAULT}:")
        for path in copied:
            printt(f"{COL_DIM}- {path}")
        printt()

    @staticmethod
    def print_missing_supporting_files_warning(missing_paths: list[str]) -> None:

        if not missing_paths:
            return

        printt(
            f"{COL_ERROR}The following supporting project files do not exist and were not copied over:{COL_DEFAULT}"
        )
        for path in missing_paths:
            printt(f"- {path}")
        printt()


ABR_OLD_VERSION_MESSAGE = (
    "Audio file was generated with an old version of tts-audiobook-tool (pre-5-2026)\n"
    "which does not contain project snapshot data."
)