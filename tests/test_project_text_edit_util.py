from pathlib import Path

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.book_serialization import (
    load_book_from_project_text_file,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
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
