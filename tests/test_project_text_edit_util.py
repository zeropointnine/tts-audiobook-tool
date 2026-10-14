import json
from pathlib import Path

import pytest

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.book_serialization import (
    load_book_from_project_text_file,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_edit_util import (
    ProjectTextEditUtil,
)
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.text_ops.text_edit_session import TextEditSession


def make_phrase_group(text: str) -> PhraseGroup:
    return PhraseGroup([Phrase(text, Reason.SENTENCE)])


def make_project(tmp_path: Path) -> Project:
    return Project.model_validate(
        {
            "dir_path": str(tmp_path),
            "book": Book(
                sections=[
                    BookSection(
                        title="Section",
                        phrase_groups=[
                            make_phrase_group("A."),
                            make_phrase_group("B."),
                            make_phrase_group("C."),
                        ],
                    )
                ]
            ),
        }
    )


def stop_project(project: Project) -> None:
    project.kill()
    project.sound_segments.observer.join(timeout=2)


def make_generated_files(project: Project) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for index, phrase_group in enumerate(project.phrase_groups):
        segment_hash = SoundSegmentUtil.calc_segment_hash(index, phrase_group.text)
        sound_path = Path(project.sound_segments_path) / (
            f"[{index + 1:05d}] [{segment_hash}] [vibevoice] [voice] line.flac"
        )
        sidecar_path = sound_path.with_suffix(".json")
        sound_path.write_bytes(b"audio")
        sidecar_path.write_text("{}", encoding="utf-8")
        result.append((sound_path, sidecar_path))
    return result


def test_commit_atomically_saves_book_and_deletes_audio_from_cutoff(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path)
    try:
        generated_files = make_generated_files(project)
        session = TextEditSession(project.book)
        session.delete_phrase_groups({session.phrase_groups[1].item_id})

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            session.earliest_affected_original_index,
        )

        assert error == ""
        assert [phrase_group.text for phrase_group in project.phrase_groups] == [
            "A.",
            "C.",
        ]
        loaded_book = load_book_from_project_text_file(project.project_text_path)
        assert isinstance(loaded_book, Book)
        assert [phrase_group.text for phrase_group in loaded_book.phrase_groups] == [
            "A.",
            "C.",
        ]
        assert all(path.exists() for path in generated_files[0])
        assert all(
            not path.exists()
            for pair in generated_files[1:]
            for path in pair
        )
    finally:
        stop_project(project)


def test_commit_rejects_stale_project_before_writing_or_deleting(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path)
    try:
        generated_files = make_generated_files(project)
        session = TextEditSession(project.book)
        session.delete_phrase_groups({session.phrase_groups[1].item_id})
        project.book.sections[0].title = "Changed elsewhere"

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            session.earliest_affected_original_index,
        )

        assert error == "Project text changed while editing"
        assert not Path(project.project_text_path).exists()
        assert all(path.exists() for pair in generated_files for path in pair)
        assert project.markers == set()
        assert not Path(project.dir_path, PROJECT_JSON_FILE_NAME).exists()
    finally:
        stop_project(project)


def test_commit_save_failure_leaves_project_and_audio_untouched(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = make_project(tmp_path)
    try:
        generated_files = make_generated_files(project)
        session = TextEditSession(project.book)
        session.delete_phrase_groups({session.phrase_groups[1].item_id})
        monkeypatch.setattr(
            ProjectTextEditUtil,
            "atomic_save_book",
            staticmethod(lambda _path, _book: "disk full"),
        )

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            session.earliest_affected_original_index,
        )

        assert error == "disk full"
        assert [phrase_group.text for phrase_group in project.phrase_groups] == [
            "A.",
            "B.",
            "C.",
        ]
        assert all(path.exists() for pair in generated_files for path in pair)
    finally:
        stop_project(project)


def test_commit_prunes_stale_section_markers_and_saves_project_json(
    tmp_path: Path,
) -> None:
    L.init("test-project-text-edit-util")
    project = make_project(tmp_path)
    try:
        # The markers setter discards non-positive values, so 0 never survives.
        project.markers = {0, 1, 2, 3}
        assert project.markers == {1, 2, 3}
        session = TextEditSession(project.book)
        session.delete_phrase_groups({session.phrase_groups[2].item_id})

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            session.earliest_affected_original_index,
        )

        assert error == ""
        assert project.markers == {1}
        on_disk = json.loads(
            (tmp_path / PROJECT_JSON_FILE_NAME).read_text(encoding="utf-8")
        )
        assert on_disk["markers"] == [1]
    finally:
        stop_project(project)


def test_commit_skips_project_save_when_no_markers_are_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = make_project(tmp_path)
    save_calls = 0

    def record_save(self) -> str:
        nonlocal save_calls
        save_calls += 1
        return ""

    monkeypatch.setattr(Project, "save", record_save)
    try:
        project.markers = {1}
        session = TextEditSession(project.book)
        session.delete_phrase_groups({session.phrase_groups[2].item_id})

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            session.earliest_affected_original_index,
        )

        assert error == ""
        assert project.markers == {1}
        assert save_calls == 0
        assert not (tmp_path / PROJECT_JSON_FILE_NAME).exists()
    finally:
        stop_project(project)


def test_commit_reports_marker_save_failure(tmp_path: Path, monkeypatch) -> None:
    project = make_project(tmp_path)
    try:
        project.markers = {1, 2, 3}
        session = TextEditSession(project.book)
        session.delete_phrase_groups({session.phrase_groups[2].item_id})
        monkeypatch.setattr(Project, "save", lambda self: "markers disk full")

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            session.earliest_affected_original_index,
        )

        assert error == (
            "Project text was saved, but section markers could not "
            "be saved: markers disk full"
        )
        assert project.markers == {1}
    finally:
        stop_project(project)


@pytest.mark.parametrize(
    ("original_text", "original_reason", "new_text", "expected_reason"),
    [
        ("Arrival\n\n", Reason.HEADING_END, "Ordinary prose.", Reason.PARAGRAPH),
        ("Ordinary prose.\n\n", Reason.PARAGRAPH, "Arrival", Reason.HEADING_END),
    ],
)
def test_confirmed_edit_recomputes_and_persists_heading_boundaries(
    tmp_path: Path, original_text, original_reason, new_text, expected_reason,
) -> None:
    # Confirmed edits can both lose and gain a heading; disk and memory must agree.
    project = make_project(tmp_path)
    try:
        project.book.sections[0].title = "Chapter 1: Arrival"
        project.book.segmentation_settings = BookSegmentationSettings(language_code="en")
        original = project.phrase_groups[0].phrases[0]
        original.text = original_text
        original.reason = original_reason
        generated_files = make_generated_files(project)
        session = TextEditSession(project.book)
        session.update_phrase_group_text(
            session.phrase_groups[0].item_id, new_text, max_words=40, pysbd_lang="en",
        )
        staged_book = session.to_book()
        error = ProjectTextEditUtil.commit(
            project, staged_book, session.original_snapshot,
            edited_segment_indices=session.edited_original_indices,
        )

        assert error == ""
        assert project.phrase_groups[0].last_reason == expected_reason
        loaded = load_book_from_project_text_file(project.project_text_path)
        assert isinstance(loaded, Book)
        assert loaded.phrase_groups[0].last_reason == expected_reason
        # The classifier acts on a detached copy, not the editor's staged objects.
        assert staged_book.phrase_groups[0].last_reason == Reason.PARAGRAPH
        assert all(not path.exists() for path in generated_files[0])
        assert all(path.exists() for pair in generated_files[1:] for path in pair)
    finally:
        stop_project(project)


def test_confirmed_edit_recomputes_untouched_sections_globally(tmp_path: Path) -> None:
    # A save elsewhere must also remove obsolete imported heading classifications.
    project = make_project(tmp_path)
    try:
        project.book.sections[0].phrase_groups[0].phrases[0].reason = Reason.HEADING_END
        session = TextEditSession(project.book)
        session.update_phrase_group_text(
            session.phrase_groups[1].item_id, "Edited prose.", max_words=40, pysbd_lang="en",
        )
        error = ProjectTextEditUtil.commit(
            project, session.to_book(), session.original_snapshot,
            edited_segment_indices=session.edited_original_indices,
        )
        assert error == ""
        assert project.phrase_groups[0].last_reason == Reason.SENTENCE
        loaded = load_book_from_project_text_file(project.project_text_path)
        assert isinstance(loaded, Book)
        assert loaded.phrase_groups[0].last_reason == Reason.SENTENCE
    finally:
        stop_project(project)


def test_commit_edit_deletes_only_edited_segment_and_keeps_markers(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path)
    try:
        generated_files = make_generated_files(project)
        project.markers = {1, 2}
        session = TextEditSession(project.book)
        session.update_phrase_group_text(
            session.phrase_groups[1].item_id,
            "B2.",
            max_words=10,
            pysbd_lang="en",
        )

        error = ProjectTextEditUtil.commit(
            project,
            session.to_book(),
            session.original_snapshot,
            earliest_affected_original_index=None,
            edited_segment_indices=session.edited_original_indices,
        )

        assert error == ""
        assert [phrase_group.text for phrase_group in project.phrase_groups] == [
            "A.",
            "B2.",
            "C.",
        ]
        assert all(path.exists() for path in generated_files[0])
        assert all(not path.exists() for path in generated_files[1])
        assert all(path.exists() for path in generated_files[2])
        assert project.markers == {1, 2}
        loaded_book = load_book_from_project_text_file(project.project_text_path)
        assert isinstance(loaded_book, Book)
        assert [phrase_group.text for phrase_group in loaded_book.phrase_groups] == [
            "A.",
            "B2.",
            "C.",
        ]
    finally:
        stop_project(project)

