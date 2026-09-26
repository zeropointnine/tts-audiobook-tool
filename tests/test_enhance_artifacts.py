from __future__ import annotations

from pathlib import Path
import json
import pickle

import pytest

from tts_audiobook_tool.app_types import (
    Book,
    BookSection,
    BookSegmentationSettings,
    ConcreteWord,
)
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.enhance import enhance_artifacts
from tts_audiobook_tool.enhance.enhance_artifacts import (
    EnhanceArtifacts,
    atomic_pickle_save,
    count_misalignments,
    delete_temporary_files,
    load_book,
    load_timed_phrases,
    load_transcription,
    make_enhance_state,
    make_enhance_state_cached,
    invalidate_state_cache,
    save_book,
    save_timed_phrases,
    save_transcription,
)


def make_book(source_kind: str = "plain_text") -> Book:
    return Book(
        sections=[BookSection([PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])], "One")],
        title="Test",
        text_source_kind=source_kind,
        audio_source_kind="pre_existing",
        segmentation_settings=BookSegmentationSettings(language_code="en", max_words_per_segment=55),
    )


def test_exact_sibling_paths_preserve_dotted_stem_and_case(tmp_path: Path) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "Novel.Part1.FLAC")

    assert artifacts.book_path == tmp_path / "Novel.Part1.abr.json"
    assert artifacts.transcription_path == tmp_path / "Novel.Part1.transcription.bin"
    assert artifacts.timed_phrases_path == tmp_path / "Novel.Part1.timed_phrases.bin"
    assert artifacts.flat_output_path == tmp_path / "Novel.Part1.abr.m4a"
    assert artifacts.epub_output_path == tmp_path / "Novel.Part1.abr.m4b"


def test_output_path_is_selected_only_from_source_kind(tmp_path: Path) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "book.mp3")

    assert artifacts.expected_output_path(make_book("plain_text")) == tmp_path / "book.abr.m4a"
    assert artifacts.expected_output_path(make_book("epub")) == tmp_path / "book.abr.m4b"
    with pytest.raises(ValueError, match="Unsupported enhance text source kind"):
        artifacts.expected_output_path(make_book("legacy_flat"))


def test_book_round_trip_uses_canonical_book_v2(tmp_path: Path) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "book.m4b")
    book = make_book("epub")

    assert save_book(artifacts, book) == ""
    loaded, error = load_book(artifacts)

    assert error == ""
    assert loaded is not None
    assert loaded.title == "Test"
    assert loaded.text_source_kind == "epub"
    assert loaded.audio_source_kind == "pre_existing"
    assert loaded.sections[0].title == "One"
    assert loaded.sections[0].phrase_groups[0].phrases[0].text == "Hello."


def test_book_load_rejects_corrupt_and_unsupported_payloads(tmp_path: Path) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "book.m4b")
    artifacts.book_path.write_text("{bad", encoding="utf-8")
    loaded, error = load_book(artifacts)
    assert loaded is None
    assert "Error loading source book" in error

    artifacts.book_path.write_text('{"format": "book.v1", "book": {}}', encoding="utf-8")
    loaded, error = load_book(artifacts)
    assert loaded is None
    assert "book.v2" in error

    assert save_book(artifacts, make_book("legacy_flat")) == ""
    loaded, error = load_book(artifacts)
    assert loaded is None
    assert "unsupported text_source_kind" in error


def test_malformed_book_shape_degrades_to_invalid_state(tmp_path: Path) -> None:
    audio = tmp_path / "book.m4b"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    artifacts.book_path.write_text(
        json.dumps({
            "format": "book.v2",
            "book": {
                "text_source_kind": "epub",
                "audio_source_kind": "pre_existing",
                "sections": [{
                    "phrase_groups": [{
                        "phrases": [{"text": 42, "reason": "s"}],
                    }],
                }],
            },
        }),
        encoding="utf-8",
    )

    state = make_enhance_state(str(audio))

    assert state.book is None
    assert "Error parsing source book" in state.book_error


def test_pickle_artifacts_round_trip_and_validate_items(tmp_path: Path) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "book.mp3")
    words = [ConcreteWord(0.0, 0.5, "hello", 0.9)]
    timed = [TimedPhrase("Hello.", 0.0, 0.5)]

    assert save_transcription(artifacts, words) == ""
    assert save_timed_phrases(artifacts, timed) == ""
    loaded_words, word_error = load_transcription(artifacts)
    loaded_timed, timed_error = load_timed_phrases(artifacts)

    assert word_error == ""
    assert loaded_words is not None and loaded_words[0].word == "hello"
    assert timed_error == ""
    assert loaded_timed is not None and loaded_timed[0].text == "Hello."

    artifacts.transcription_path.write_bytes(pickle.dumps({"not": "a list"}))
    assert load_transcription(artifacts)[0] is None
    assert "expected a list" in load_transcription(artifacts)[1]
    artifacts.timed_phrases_path.write_bytes(pickle.dumps([object()]))
    assert load_timed_phrases(artifacts)[0] is None
    assert "item at index 0" in load_timed_phrases(artifacts)[1]


def test_atomic_pickle_failure_preserves_previous_destination(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "cache.bin"
    destination.write_bytes(b"known-good")

    def fail_dump(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("tts_audiobook_tool.enhance.enhance_artifacts.pickle.dump", fail_dump)
    error = atomic_pickle_save(destination, ["replacement"])

    assert "disk full" in error
    assert destination.read_bytes() == b"known-good"
    assert list(tmp_path.glob(".cache.bin.*.tmp")) == []


def test_misalignment_count_counts_records_not_ranges() -> None:
    phrases = [
        TimedPhrase("A", 0.0, 0.0),
        TimedPhrase("B", 0.0, 0.0),
        TimedPhrase("C", 1.0, 2.0),
        TimedPhrase("D", 0.0, 0.0),
    ]
    assert count_misalignments(phrases) == 3


def test_cleanup_removes_only_three_work_files(tmp_path: Path) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "book.mp3")
    preserved = [artifacts.audio_path, artifacts.flat_output_path, artifacts.epub_output_path]
    for path in (*artifacts.temporary_paths, *preserved):
        path.write_bytes(b"x")

    assert delete_temporary_files(artifacts) == []
    assert all(not path.exists() for path in artifacts.temporary_paths)
    assert all(path.exists() for path in preserved)
    assert delete_temporary_files(artifacts) == []


def test_cleanup_aggregates_permission_errors_and_continues(tmp_path: Path, monkeypatch) -> None:
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "book.mp3")
    for path in artifacts.temporary_paths:
        path.write_bytes(b"x")
    real_unlink = Path.unlink

    def selective_unlink(path: Path, *args, **kwargs):
        if path == artifacts.transcription_path:
            raise PermissionError("denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", selective_unlink)
    errors = delete_temporary_files(artifacts)

    assert len(errors) == 1
    assert str(artifacts.transcription_path) in errors[0]
    assert not artifacts.book_path.exists()
    assert artifacts.transcription_path.exists()
    assert not artifacts.timed_phrases_path.exists()


def test_state_uses_valid_output_metadata_as_review_fallback(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "book.m4b"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    assert save_book(artifacts, make_book("epub")) == ""
    assert save_transcription(artifacts, [ConcreteWord(0.0, 0.5, "hello", 1.0)]) == ""
    artifacts.epub_output_path.write_bytes(b"output")
    metadata = AppMetadata(
        timed_phrases=[TimedPhrase("A", 0.0, 0.0), TimedPhrase("B", 1.0, 2.0)],
        title="Test",
        version=1,
        bookmark_indices=[],
        raw_text="",
        has_break_audio=False,
        project_snapshot={},
        sections=[],
    )
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: metadata))

    state = make_enhance_state(str(audio))

    assert state.audio_exists
    assert state.book is not None
    assert state.transcription_valid
    assert not state.timed_phrases_exists
    assert state.output_valid
    assert state.expected_output_path == artifacts.epub_output_path
    assert state.output_phrase_count == 2
    assert state.misalignment_count == 1


def test_state_counts_embedded_output_not_newer_timed_cache(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "book.mp3"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    assert save_book(artifacts, make_book()) == ""
    assert save_timed_phrases(artifacts, [
        TimedPhrase("New A", 0, 0), TimedPhrase("New B", 0, 0), TimedPhrase("New C", 0, 0),
    ]) == ""
    artifacts.flat_output_path.write_bytes(b"older output")
    embedded = AppMetadata(
        timed_phrases=[TimedPhrase("Old A", 0, 0), TimedPhrase("Old B", 1, 2)],
        title="Test", version=1, bookmark_indices=[], raw_text="",
        has_break_audio=False, project_snapshot={}, sections=[],
    )
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: embedded))

    state = make_enhance_state(str(audio))

    assert state.timed_phrases_valid
    assert state.output_valid
    assert state.output_phrase_count == 2
    assert state.misalignment_count == 1


def test_empty_selection_has_no_derived_artifacts() -> None:
    state = make_enhance_state("")
    assert not state.has_audio_path
    assert state.artifacts is None
    assert state.expected_output_path is None


def _cache_test_setup(tmp_path: Path) -> tuple[Path, EnhanceArtifacts]:
    audio = tmp_path / "book.mp3"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    assert save_book(artifacts, make_book()) == ""
    assert save_transcription(artifacts, [ConcreteWord(0.0, 0.5, "hello", 1.0)]) == ""
    return audio, artifacts


def test_cached_state_reuses_snapshot_until_a_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio, artifacts = _cache_test_setup(tmp_path)
    derivations = []
    real_make = make_enhance_state
    monkeypatch.setattr(
        enhance_artifacts,
        "make_enhance_state",
        lambda path: derivations.append(path) or real_make(path),
    )
    try:
        first = make_enhance_state_cached(str(audio))
        second = make_enhance_state_cached(str(audio))
        assert second is first
        assert len(derivations) == 1

        # Atomic saves bump st_mtime_ns, so the next render re-derives.
        assert save_timed_phrases(artifacts, [TimedPhrase("Hello.", 0.0, 0.5)]) == ""
        third = make_enhance_state_cached(str(audio))
        assert third is not first
        assert third.timed_phrases_valid
        assert len(derivations) == 2

        # Deleting a derived file also invalidates.
        artifacts.transcription_path.unlink()
        fourth = make_enhance_state_cached(str(audio))
        assert not fourth.transcription_exists
        assert len(derivations) == 3
    finally:
        invalidate_state_cache()


def test_cached_state_rederives_when_audio_file_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio, _artifacts = _cache_test_setup(tmp_path)
    assert make_enhance_state_cached(str(audio)).audio_exists

    audio.unlink()
    state = make_enhance_state_cached(str(audio))
    assert not state.audio_exists
    invalidate_state_cache()


def test_cached_state_never_caches_the_empty_selection() -> None:
    assert make_enhance_state_cached("").artifacts is None
    assert "" not in enhance_artifacts._state_cache
