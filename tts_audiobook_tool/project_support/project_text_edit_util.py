from collections.abc import Collection
from copy import deepcopy

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
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
        earliest_affected_original_index: int | None = None,
        edited_segment_indices: Collection[int] | None = None,
    ) -> str:
        """Persist a staged Book, update Project, then remove invalid generated audio.

        ``edited_segment_indices`` selects edit-only invalidation: only those
        sound segments are deleted and markers are left untouched. Otherwise
        ``earliest_affected_original_index`` selects the structural cutoff,
        deleting segments from that index onward and pruning markers at/after it.
        """
        if TextEditSession.make_snapshot(project.book) != original_snapshot:
            return "Project text changed while editing"
        if not project.project_text_path:
            return "Project text path is unavailable"

        if edited_segment_indices is not None:
            cleanup_paths = project.sound_segments.snapshot_paths_at_indices(
                edited_segment_indices
            )
            prune_markers_cutoff = None
        elif earliest_affected_original_index is not None:
            cleanup_paths = project.sound_segments.snapshot_paths_from_index(
                earliest_affected_original_index
            )
            prune_markers_cutoff = earliest_affected_original_index
        else:
            cleanup_paths = []
            prune_markers_cutoff = None

        new_book = deepcopy(staged_book)
        error = ProjectTextEditUtil.atomic_save_book(project.project_text_path, new_book)
        if error:
            return error

        try:
            project.book = new_book
        except Exception as exception:
            return (
                "Project text was saved, but the in-memory project update failed: "
                f"{make_error_string(exception)}"
            )

        if prune_markers_cutoff is not None:
            new_markers = {
                marker
                for marker in project.markers
                if marker < prune_markers_cutoff
            }
            if new_markers != project.markers:
                project.markers = new_markers
                err = project.save()
                if err:
                    return (
                        "Project text was saved, but section markers could not "
                        f"be saved: {err}"
                    )

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
        """Compatibility wrapper around the shared atomic project-text saver."""
        return JsonSaveUtil.save(
            JsonArtifactType.PROJECT_TEXT,
            path,
            lambda: book_to_project_text_json_dict(book),
        )
