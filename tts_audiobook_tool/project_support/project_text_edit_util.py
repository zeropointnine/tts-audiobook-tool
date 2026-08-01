from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.app_types.book_serialization import (
    book_to_project_text_json_dict,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_ops.text_edit_session import TextEditSession
from tts_audiobook_tool.util import make_error_string


class ProjectTextEditUtil:
    """Coordinates a confirmed structural text edit across memory and disk."""

    @staticmethod
    def commit(
        project: Project,
        staged_book: Book,
        original_snapshot: str,
        earliest_affected_original_index: int | None,
    ) -> str:
        """Persist a staged Book, update Project, then remove invalid generated audio."""
        if TextEditSession.make_snapshot(project.book) != original_snapshot:
            return "Project text changed while editing"
        if not project.project_text_path:
            return "Project text path is unavailable"

        cleanup_paths = (
            project.sound_segments.snapshot_paths_from_index(
                earliest_affected_original_index
            )
            if earliest_affected_original_index is not None
            else []
        )
        new_book = deepcopy(staged_book)
        error = ProjectTextEditUtil.atomic_save_book(project.project_text_path, new_book)
        if error:
            return error

        autosave = project._autosave
        try:
            project._autosave = False
            project.book = new_book
            project._phrase_groups_dirty = False
            project._phrase_groups_inline_source = ""
        except Exception as exception:
            return (
                "Project text was saved, but the in-memory project update failed: "
                f"{make_error_string(exception)}"
            )
        finally:
            project._autosave = autosave

        try:
            project.sound_segments.delete_path_snapshot(cleanup_paths)
        except Exception as exception:
            project.sound_segments.force_invalidate()
            return (
                "Project text was saved, but some outdated sound segments could not "
                f"be deleted: {make_error_string(exception)}"
            )
        return ""

    @staticmethod
    def atomic_save_book(path: str, book: Book) -> str:
        """Write and fsync a temporary sibling before atomically replacing path."""
        destination = Path(path)
        temporary_path = ""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = file.name
                json.dump(book_to_project_text_json_dict(book), file, indent=4)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, destination)
            return ""
        except Exception as exception:
            return make_error_string(exception)
        finally:
            if temporary_path:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass
