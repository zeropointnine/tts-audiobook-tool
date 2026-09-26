from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tts_audiobook_tool.app_types import (
    Book,
    BookSection,
    BookSegmentationSettings,
    ConcreteWord,
    SegmentationStrategy,
)
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.enhance import enhance_flow
from tts_audiobook_tool.textual.content_textual_app import ContentAppCompleted, ContentAppUnavailable, EditorClosed
from tts_audiobook_tool.enhance.enhance_artifacts import (
    EnhanceArtifacts,
    load_transcription,
    save_book,
    save_timed_phrases,
    save_transcription,
)
from tts_audiobook_tool.prefs import Prefs


def make_book(source_kind: str = "plain_text") -> Book:
    return Book(
        sections=[BookSection([PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])], "One")],
        title="Book",
        text_source_kind=source_kind,
        audio_source_kind="pre_existing",
        segmentation_settings=BookSegmentationSettings(),
    )


def make_state(audio_path: Path | None = None):
    prefs = Prefs(hints={}, enhance_audio_path=str(audio_path) if audio_path else "")
    project = SimpleNamespace(
        max_words=55,
        segmentation_strategy=SegmentationStrategy.SENTENCE_PLUS,
        language_code="en",
        dialog_segmentation=False,
    )
    return SimpleNamespace(prefs=prefs, project=project)


def prepare(tmp_path: Path, source_kind="plain_text"):
    audio = tmp_path / "novel.mp3"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    assert save_book(artifacts, make_book(source_kind)) == ""
    state = make_state(audio)
    return state, artifacts


def silence_ui(monkeypatch):
    monkeypatch.setattr(enhance_flow, "printt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(enhance_flow.MenuUtil, "print_heading", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(enhance_flow, "print_feedback", lambda *_args, **_kwargs: None)


def test_select_audio_validates_then_saves_one_normalized_selection(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "old.m4b"
    selected = tmp_path / "sub" / ".." / "selected.mp3"
    normalized = tmp_path / "selected.mp3"
    normalized.write_bytes(b"audio")
    state = make_state(old)
    saves: list[str] = []
    monkeypatch.setattr(enhance_flow.ask, "ask_file_path", lambda *_args, **_kwargs: str(selected))
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: None))
    monkeypatch.setattr(state.prefs, "save", lambda: saves.append(state.prefs.enhance_audio_path) or "")

    enhance_flow.select_audio(state)

    assert state.prefs.enhance_audio_path == str(normalized)
    assert saves == [str(normalized)]


@pytest.mark.parametrize("clear_first", [False, True])
def test_select_audio_preserves_existing_work_files(tmp_path: Path, monkeypatch, clear_first: bool) -> None:
    audio = tmp_path / "novel.mp3"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    assert save_book(artifacts, make_book()) == ""
    assert save_transcription(artifacts, [ConcreteWord(0, 0.5, "hello", 1)]) == ""
    assert save_timed_phrases(artifacts, [TimedPhrase("Hello.", 0, 0.5)]) == ""
    original_files = {path: path.read_bytes() for path in artifacts.temporary_paths}

    # Both switching from another book and reselecting after Clear must resume
    # the work already stored alongside the newly selected audio file.
    state = make_state(audio if clear_first else tmp_path / "other.mp3")
    monkeypatch.setattr(state.prefs, "save", lambda: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: False)
    monkeypatch.setattr(enhance_flow.ask, "ask_file_path", lambda *_args, **_kwargs: str(audio))
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: None))
    if clear_first:
        enhance_flow.clear_selection(state)
        assert state.prefs.enhance_audio_path == ""
        assert all(path.read_bytes() == contents for path, contents in original_files.items())

    enhance_flow.select_audio(state)

    assert state.prefs.enhance_audio_path == str(audio)
    assert all(path.read_bytes() == contents for path, contents in original_files.items())


def test_select_audio_notifies_when_parallel_output_already_exists(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "novel.mp3"
    audio.write_bytes(b"audio")
    artifacts = EnhanceArtifacts.from_audio_path(audio)
    artifacts.epub_output_path.write_bytes(b"existing output")  # suffix unknowable pre-book
    shown: list[tuple[object, bool]] = []
    state = make_state(audio)
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: None))
    monkeypatch.setattr(enhance_flow.ask, "ask_file_path", lambda *_args, **_kwargs: str(audio))
    monkeypatch.setattr(
        enhance_flow.hints, "show_hint", lambda hint, **kwargs: shown.append((hint, kwargs.get("and_prompt")))
    )
    monkeypatch.setattr(state.prefs, "save", lambda: "")

    enhance_flow.select_audio(state)

    assert state.prefs.enhance_audio_path == str(audio)
    assert len(shown) == 1
    hint, and_prompt = shown[0]
    assert and_prompt is True
    assert "An enhanced audiobook already exists" in hint.heading
    assert str(artifacts.epub_output_path) in hint.heading + hint.text
    assert "review" in hint.text


def test_select_audio_save_failure_restores_previous_selection(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "old.m4b"
    selected = tmp_path / "selected.mp3"
    selected.write_bytes(b"audio")
    state = make_state(old)
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow.ask, "ask_file_path", lambda *_args, **_kwargs: str(selected))
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: None))
    monkeypatch.setattr(state.prefs, "save", lambda: "disk full")
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)

    enhance_flow.select_audio(state)

    assert state.prefs.enhance_audio_path == str(old)
    assert "disk full" in errors[0]


def test_select_text_atomically_replaces_book_then_invalidates_timed_cache(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Replacement", encoding="utf-8")
    words = [ConcreteWord(0, 0.5, "hello", 1)]
    assert save_transcription(artifacts, words) == ""
    original_transcription = artifacts.transcription_path.read_bytes()
    assert save_timed_phrases(artifacts, [TimedPhrase("old", 0, 0)]) == ""
    replacement = make_book("plain_text")
    replacement.title = "Replacement"
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(enhance_flow.ask, "ask_file_path", lambda *_args, **_kwargs: str(source))
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: True)
    monkeypatch.setattr(enhance_flow.enhance_text, "import_source_book", lambda *_args: replacement)
    monkeypatch.setattr(state.prefs, "save", lambda: "")

    enhance_flow.select_text(state)

    assert artifacts.book_path.exists()
    assert artifacts.transcription_path.read_bytes() == original_transcription
    cached_words, error = load_transcription(artifacts)
    assert error == ""
    assert cached_words is not None
    assert [(word.start, word.end, word.word, word.probability) for word in cached_words] == [
        (0, 0.5, "hello", 1)
    ]
    assert not artifacts.timed_phrases_path.exists()
    assert state.prefs.last_text_dir == str(tmp_path)


def test_failed_book_save_preserves_old_book_and_timed_cache(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Replacement", encoding="utf-8")
    original = artifacts.book_path.read_bytes()
    assert save_timed_phrases(artifacts, [TimedPhrase("old", 0, 0)]) == ""
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(enhance_flow.ask, "ask_file_path", lambda *_args, **_kwargs: str(source))
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: True)
    monkeypatch.setattr(enhance_flow.enhance_text, "import_source_book", lambda *_args: make_book())
    monkeypatch.setattr(state.prefs, "save", lambda: "")
    monkeypatch.setattr(enhance_flow, "save_book", lambda *_args: "disk full")
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)

    enhance_flow.select_text(state)

    assert artifacts.book_path.read_bytes() == original
    assert artifacts.timed_phrases_path.exists()
    assert errors == ["disk full"]


def test_transcription_heading_has_exactly_one_blank_line_before_progress(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    state, _ = prepare(tmp_path)
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: False)

    def show_first_progress(*_args):
        print("TRANSCRIPTION_START")
        return None

    monkeypatch.setattr(
        enhance_flow.enhance_alignment, "transcribe_to_words", show_first_progress
    )
    enhance_flow.transcribe(state)

    output = enhance_flow.text_util.strip_ansi_codes(capsys.readouterr().out)
    assert "Transcribing audio (This may take some time...)\n\nTRANSCRIPTION_START\n" in output


def test_transcription_success_saves_sibling_and_invalidates_timing(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    assert save_timed_phrases(artifacts, [TimedPhrase("old", 0, 0)]) == ""
    words = [ConcreteWord(0.0, 0.5, "hello", 1.0)]
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.enhance_alignment, "transcribe_to_words", lambda *_args: words)
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: False)
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.transcribe(state)

    assert artifacts.transcription_path.exists()
    assert not artifacts.timed_phrases_path.exists()


def test_transcription_prompt_chains_alignment_and_output_only_if_accepted(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "epub")
    words = [ConcreteWord(0.0, 0.5, "hello", 1.0)]
    timed = [TimedPhrase("Hello.", 0.0, 0.5), TimedPhrase("orphan", 0.0, 0.0)]
    prompts: list[str] = []
    outputs: list[Path] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.enhance_alignment, "transcribe_to_words", lambda *_args: words)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: (timed, False))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda _state, _artifacts, path, _meta: outputs.append(path) or "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda message: prompts.append(message) or True)
    monkeypatch.setattr(enhance_flow.app_hint_util, "show_player_hint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.transcribe(state)

    assert prompts == [
        'When transcription is finished, perform force-alignment step and create the ".abr.m4b" file? ',
        "Review orphaned lines now? ",
    ]
    assert artifacts.transcription_path.exists()
    assert artifacts.timed_phrases_path.exists()
    assert outputs == [artifacts.epub_output_path]


def test_transcription_declined_chain_saves_only_transcription(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    prompts: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.enhance_alignment, "transcribe_to_words", lambda *_args: [ConcreteWord(0, 0.5, "hello", 1)])
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda message: prompts.append(message) or False)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: pytest.fail("must not align"))
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.transcribe(state)

    assert prompts == ['When transcription is finished, perform force-alignment step and create the ".abr.m4a" file? ']
    assert artifacts.transcription_path.exists()
    assert not artifacts.timed_phrases_path.exists()


def test_transcription_chained_alignment_interruption_preserves_existing_output(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    artifacts.flat_output_path.write_bytes(b"old output")
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.enhance_alignment, "transcribe_to_words", lambda *_args: [ConcreteWord(0, 0.5, "hello", 1)])
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: True)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: ([], True))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.transcribe(state)

    assert artifacts.transcription_path.exists()
    assert not artifacts.timed_phrases_path.exists()
    assert artifacts.flat_output_path.read_bytes() == b"old output"


def test_transcription_without_book_transcribes_without_chain_prompt(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    artifacts.book_path.unlink()
    prompts: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.enhance_alignment, "transcribe_to_words", lambda *_args: [ConcreteWord(0, 0.5, "hello", 1)])
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda message: prompts.append(message) or pytest.fail("must not confirm"))
    monkeypatch.setattr(enhance_flow.hints, "show_hint_if_necessary", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.transcribe(state)

    assert prompts == []
    assert artifacts.transcription_path.exists()


def test_transcription_error_preserves_previous_cache(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    old = [ConcreteWord(0.0, 0.5, "old", 1.0)]
    assert save_transcription(artifacts, old) == ""
    before = artifacts.transcription_path.read_bytes()
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: True)
    monkeypatch.setattr(
        enhance_flow.enhance_alignment,
        "transcribe_to_words",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("worker died")),
    )
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)
    silence_ui(monkeypatch)

    enhance_flow.transcribe(state)

    assert artifacts.transcription_path.read_bytes() == before
    assert "worker died" in errors[0]


def metadata() -> AppMetadata:
    return AppMetadata(
        timed_phrases=[TimedPhrase("Hello", 0.0, 1.0)],
        title="Book",
        version=4,
        bookmark_indices=[],
        raw_text="",
        has_break_audio=False,
        project_snapshot={},
        sections=[],
    )


def test_explicit_transcode_helper_uses_default_audio_and_requested_bitrate(monkeypatch) -> None:
    captured = {}

    def make_file(command, destination, use_temp_file):
        captured.update(command=command, destination=destination, use_temp_file=use_temp_file)
        return ""

    monkeypatch.setattr(
        "tts_audiobook_tool.sound.sound_file_util.FfmpegUtil.make_file",
        make_file,
    )
    error = enhance_flow.SoundFileUtil.transcode_to_aac_at(
        "source.mp4",
        "staging.m4b",
        "128k",
    )

    assert error == ""
    assert captured["destination"] == "staging.m4b"
    assert captured["use_temp_file"] is True
    command = captured["command"]
    assert "-map" not in command  # FFmpeg picks the same default track as STT.
    assert command[command.index("-b:a") + 1] == "128k"
    assert all(option in command for option in ("-vn", "-sn", "-dn"))


def test_staged_output_copies_m4a_without_reencoding(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.m4a"
    source.write_bytes(b"source")
    artifacts = EnhanceArtifacts.from_audio_path(source)
    destination = artifacts.flat_output_path
    state = make_state(source)
    transcode = Mock(return_value="")
    monkeypatch.setattr(enhance_flow.SoundFileUtil, "transcode_to_aac_at", transcode)

    def save_meta(_meta, src, dest=""):
        Path(dest).write_bytes(Path(src).read_bytes() + b"+meta")
        return ""

    monkeypatch.setattr(AppMetadata, "save_to_mp4", staticmethod(save_meta))
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: metadata()))

    assert enhance_flow._write_staged_output(state, artifacts, destination, metadata()) == ""
    assert destination.read_bytes() == b"source+meta"
    transcode.assert_not_called()


def test_staged_output_transcodes_mp3_with_bitrate_and_strips_via_helper(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.mp3"
    source.write_bytes(b"source")
    artifacts = EnhanceArtifacts.from_audio_path(source)
    destination = artifacts.epub_output_path
    state = make_state(source)
    calls: list[tuple[str, str, str]] = []

    def transcode(src, dest, bitrate):
        calls.append((src, dest, bitrate))
        Path(dest).write_bytes(b"aac")
        return ""

    monkeypatch.setattr(enhance_flow.SoundFileUtil, "transcode_to_aac_at", transcode)
    monkeypatch.setattr(AppMetadata, "save_to_mp4", staticmethod(lambda *_args, **_kwargs: ""))
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: metadata()))

    assert enhance_flow._write_staged_output(state, artifacts, destination, metadata()) == ""
    assert destination.read_bytes() == b"aac"
    assert calls[0][0] == str(source)
    assert calls[0][2] == state.prefs.aac_bitrate


def test_failed_staging_preserves_existing_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.mp3"
    source.write_bytes(b"source")
    artifacts = EnhanceArtifacts.from_audio_path(source)
    destination = artifacts.flat_output_path
    destination.write_bytes(b"known-good")
    state = make_state(source)
    monkeypatch.setattr(
        enhance_flow.SoundFileUtil,
        "transcode_to_aac_at",
        lambda *_args: "ffmpeg failed",
    )

    assert "ffmpeg failed" in enhance_flow._write_staged_output(state, artifacts, destination, metadata())
    assert destination.read_bytes() == b"known-good"
    assert list(tmp_path.glob(".book.abr.*.m4a")) == []


def test_metadata_exception_preserves_existing_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.m4a"
    source.write_bytes(b"source")
    artifacts = EnhanceArtifacts.from_audio_path(source)
    destination = artifacts.flat_output_path
    destination.write_bytes(b"known-good")
    state = make_state(source)
    monkeypatch.setattr(
        AppMetadata,
        "save_to_mp4",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("tag failed"))),
    )

    error = enhance_flow._write_staged_output(state, artifacts, destination, metadata())

    assert "tag failed" in error
    assert destination.read_bytes() == b"known-good"


def test_final_replace_failure_preserves_existing_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "book.m4a"
    source.write_bytes(b"source")
    artifacts = EnhanceArtifacts.from_audio_path(source)
    destination = artifacts.flat_output_path
    destination.write_bytes(b"known-good")
    state = make_state(source)

    def save_meta(_meta, src, dest=""):
        Path(dest).write_bytes(Path(src).read_bytes() + b"+meta")
        return ""

    monkeypatch.setattr(AppMetadata, "save_to_mp4", staticmethod(save_meta))
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: metadata()))
    monkeypatch.setattr(
        enhance_flow.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("rename failed")),
    )

    error = enhance_flow._write_staged_output(state, artifacts, destination, metadata())

    assert "rename failed" in error
    assert destination.read_bytes() == b"known-good"


def test_alignment_requires_transcription_before_prompting(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: pytest.fail("must not confirm"))
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: pytest.fail("must not align"))

    enhance_flow.align_source_text(state)

    assert errors and "transcription" in errors[0]
    assert not artifacts.timed_phrases_path.exists()


def test_manual_alignment_declines_output_and_saves_timed_phrases(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    assert save_transcription(artifacts, [ConcreteWord(0, 0.5, "hello", 1)]) == ""
    prompts: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda message: prompts.append(message) or False)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: ([TimedPhrase("Hello.", 0, 0.5)], False))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))
    silence_ui(monkeypatch)

    enhance_flow.align_source_text(state)

    assert prompts == ['Create the ".abr.m4a" file when alignment is finished? ']
    assert artifacts.timed_phrases_path.exists()
    assert not artifacts.flat_output_path.exists()


def test_alignment_without_create_still_prompts_orphan_review_after_save(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    assert save_transcription(artifacts, [ConcreteWord(0, 0.5, "hello", 1)]) == ""
    prompts: list[str] = []
    reviewed: list[bool] = []
    timed = [TimedPhrase("Hello.", 0.0, 0.5), TimedPhrase("orphan", 0.0, 0.0)]
    def confirm(message: str) -> bool:
        prompts.append(message)
        return message.startswith("Review")  # decline create, accept review

    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", confirm)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: (timed, False))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))
    monkeypatch.setattr(enhance_flow, "review_discontinuities", lambda _state: reviewed.append(True))
    silence_ui(monkeypatch)

    enhance_flow.align_source_text(state)

    assert prompts == ['Create the ".abr.m4a" file when alignment is finished? ', "Review orphaned lines now? "]
    assert artifacts.timed_phrases_path.exists()
    assert reviewed == [True]


def test_manual_alignment_yes_creates_epub_output(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "epub")
    assert save_transcription(artifacts, [ConcreteWord(0, 0.5, "hello", 1)]) == ""
    prompts: list[str] = []
    outputs: list[Path] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda message: prompts.append(message) or True)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: ([TimedPhrase("Hello.", 0, 0.5)], False))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda _state, _artifacts, path, _meta: outputs.append(path) or "")
    monkeypatch.setattr(enhance_flow.app_hint_util, "show_player_hint", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.align_source_text(state)

    # No orphans in the alignment, so no orphan-review prompt.
    assert prompts == ['Create the ".abr.m4b" file when alignment is finished? ']
    assert artifacts.timed_phrases_path.exists()
    assert outputs == [artifacts.epub_output_path]


def test_create_uses_saved_alignment_and_section_metadata_without_realigning(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "plain_text")
    timed = [TimedPhrase("Hello.", 0.0, 0.5)]
    assert save_timed_phrases(artifacts, timed) == ""
    captured: list[tuple[Path, AppMetadata]] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: pytest.fail("must not realign"))
    monkeypatch.setattr(
        enhance_flow,
        "_write_staged_output",
        lambda _state, _artifacts, path, meta: captured.append((path, meta)) or "",
    )
    monkeypatch.setattr(enhance_flow.app_hint_util, "show_player_hint", lambda *_args, **_kwargs: None)
    silence_ui(monkeypatch)

    enhance_flow.create_output(state)

    assert artifacts.timed_phrases_path.exists()
    assert captured[0][0] == artifacts.flat_output_path
    assert captured[0][1].raw_text == ""
    assert [(item.start_index, item.end_index) for item in captured[0][1].sections] == [(0, 1)]


def test_align_interruption_preserves_prior_cache_and_does_not_create_output(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "epub")
    assert save_transcription(artifacts, [ConcreteWord(0.0, 0.5, "hello", 1.0)]) == ""
    old = [TimedPhrase("Old", 0, 0.2)]
    assert save_timed_phrases(artifacts, old) == ""
    before = artifacts.timed_phrases_path.read_bytes()
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: True)
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: ([], True))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))
    silence_ui(monkeypatch)

    enhance_flow.align_source_text(state)

    assert artifacts.timed_phrases_path.read_bytes() == before
    assert not artifacts.epub_output_path.exists()


def test_create_with_missing_book_and_cached_alignment_explains_stale_state(
    tmp_path: Path, monkeypatch
) -> None:
    state, artifacts = prepare(tmp_path, "plain_text")
    artifacts.book_path.unlink()
    assert save_timed_phrases(artifacts, [TimedPhrase("Hello.", 0.0, 0.5)]) == ""
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: pytest.fail("must not confirm"))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))

    enhance_flow.create_output(state)

    assert errors == [
        "Source text required. The alignment was built from a book that is no "
        "longer present; re-enter the source text (its alignment will be redone)."
    ]
    assert not artifacts.flat_output_path.exists()


@pytest.mark.parametrize("cache", ["missing", "corrupt", "empty"])
def test_create_rejects_invalid_alignment_without_touching_output(tmp_path: Path, monkeypatch, cache: str) -> None:
    state, artifacts = prepare(tmp_path)
    artifacts.flat_output_path.write_bytes(b"old output")
    if cache == "corrupt":
        artifacts.timed_phrases_path.write_bytes(b"not pickle")
    elif cache == "empty":
        assert save_timed_phrases(artifacts, []) == ""
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: pytest.fail("must not confirm"))
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: pytest.fail("must not align"))
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))

    enhance_flow.create_output(state)

    assert errors and ("timed" in errors[0] or "Alignment" in errors[0])
    assert artifacts.flat_output_path.read_bytes() == b"old output"


def test_create_existing_output_requires_confirmation(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    assert save_timed_phrases(artifacts, [TimedPhrase("Hello.", 0, 0.5)]) == ""
    artifacts.flat_output_path.write_bytes(b"old output")
    confirmations: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda message: confirmations.append(message) or False)
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))

    enhance_flow.create_output(state)

    assert confirmations == [f"Replace the existing enhanced audiobook at {artifacts.flat_output_path}? "]
    assert artifacts.flat_output_path.read_bytes() == b"old output"


def test_align_failure_and_save_failure_preserve_existing_cache(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    assert save_transcription(artifacts, [ConcreteWord(0, 0.5, "hello", 1)]) == ""
    assert save_timed_phrases(artifacts, [TimedPhrase("old", 0, 0.5)]) == ""
    original = artifacts.timed_phrases_path.read_bytes()
    artifacts.flat_output_path.write_bytes(b"old output")
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow, "_validate_audio", lambda _path: "")
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", lambda *_args: True)
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)
    monkeypatch.setattr(enhance_flow, "_write_staged_output", lambda *_args: pytest.fail("must not write"))
    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: (_ for _ in ()).throw(RuntimeError("align failed")))
    silence_ui(monkeypatch)

    enhance_flow.align_source_text(state)
    assert "align failed" in errors.pop()
    assert artifacts.timed_phrases_path.read_bytes() == original

    monkeypatch.setattr(enhance_flow.enhance_text, "align_book", lambda *_args: ([TimedPhrase("new", 0, 0.5)], False))
    monkeypatch.setattr(enhance_flow, "save_timed_phrases", lambda *_args: "disk full")
    enhance_flow.align_source_text(state)
    assert errors == ["disk full"]
    assert artifacts.timed_phrases_path.read_bytes() == original
    assert artifacts.flat_output_path.read_bytes() == b"old output"


def test_review_uses_embedded_phrases_even_with_newer_timed_cache(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "plain_text")
    artifacts.flat_output_path.write_bytes(b"final")
    cached = [TimedPhrase("New text", 0, 0), TimedPhrase("Extra line", 0, 0)]
    assert save_timed_phrases(artifacts, cached) == ""
    embedded = metadata()
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: embedded))
    monkeypatch.setattr(
        enhance_flow.AudioMetaUtil, "get_audio_duration",
        lambda path: 123.0 if path == str(artifacts.flat_output_path) else pytest.fail("wrong audio path"),
    )
    reviewed = []
    monkeypatch.setattr(enhance_flow, "run_content_textual_app", lambda app: reviewed.append(app) or ContentAppCompleted(EditorClosed()))
    monkeypatch.setattr(enhance_flow.ask, "ask_enter_to_continue", lambda: pytest.fail("must not pause separately"))

    enhance_flow.review_discontinuities(state)

    assert len(reviewed) == 1
    assert reviewed[0].project is state.project
    assert reviewed[0].audio_path == artifacts.flat_output_path
    assert reviewed[0].audio_duration == 123.0
    assert reviewed[0].timed_phrases == embedded.timed_phrases


def test_review_falls_back_to_embedded_output_metadata(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "plain_text")
    artifacts.flat_output_path.write_bytes(b"final")
    embedded = metadata()
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: embedded))
    monkeypatch.setattr(enhance_flow.AudioMetaUtil, "get_audio_duration", lambda _path: 65.0)
    reviewed = []
    monkeypatch.setattr(enhance_flow, "run_content_textual_app", lambda app: reviewed.append(app) or ContentAppCompleted(EditorClosed()))

    enhance_flow.review_discontinuities(state)

    assert len(reviewed) == 1
    assert reviewed[0].audio_path == artifacts.flat_output_path
    assert reviewed[0].audio_duration == 65.0
    assert reviewed[0].timed_phrases == embedded.timed_phrases


def test_review_reports_textual_unavailable(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    artifacts.flat_output_path.write_bytes(b"final")
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: metadata()))
    monkeypatch.setattr(enhance_flow.AudioMetaUtil, "get_audio_duration", lambda _path: None)
    monkeypatch.setattr(enhance_flow, "run_content_textual_app", lambda app: ContentAppUnavailable("No terminal"))
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)

    enhance_flow.review_discontinuities(state)

    assert errors == ["No terminal"]


def test_review_reports_unreadable_output_without_pausing(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "plain_text")
    artifacts.flat_output_path.write_bytes(b"not metadata")
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: None))
    errors: list[str] = []
    monkeypatch.setattr(enhance_flow.ask, "ask_error", errors.append)
    monkeypatch.setattr(
        enhance_flow.ask,
        "ask_enter_to_continue",
        lambda: (_ for _ in ()).throw(AssertionError("must not pause separately")),
    )

    enhance_flow.review_discontinuities(state)

    assert errors and "metadata" in errors[0]


def test_clear_with_output_deletes_only_work_files_and_saves_empty_path(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path, "plain_text")
    artifacts.audio_path.write_bytes(b"audio")
    artifacts.flat_output_path.write_bytes(b"final")
    assert save_transcription(artifacts, []) == ""
    assert save_timed_phrases(artifacts, []) == ""
    monkeypatch.setattr(AppMetadata, "load_from_file", staticmethod(lambda _path: metadata()))
    confirmations: list[str] = []
    monkeypatch.setattr(
        enhance_flow.ask,
        "ask_confirm",
        lambda message: confirmations.append(message) or True,
    )
    saved: list[str] = []
    monkeypatch.setattr(state.prefs, "save", lambda: saved.append(state.prefs.enhance_audio_path) or "")

    enhance_flow.clear_selection(state)

    assert confirmations == ["Also delete the intermediate files created by this workflow?"]
    assert all(not path.exists() for path in artifacts.temporary_paths)
    assert artifacts.audio_path.exists()
    assert artifacts.flat_output_path.exists()
    assert state.prefs.enhance_audio_path == ""
    assert saved == [""]


def test_clear_without_output_still_prompts_when_work_files_exist(tmp_path: Path, monkeypatch) -> None:
    state, artifacts = prepare(tmp_path)
    confirmations: list[str] = []
    monkeypatch.setattr(
        enhance_flow.ask,
        "ask_confirm",
        lambda message: confirmations.append(message) or True,
    )
    monkeypatch.setattr(state.prefs, "save", lambda: "")

    enhance_flow.clear_selection(state)

    assert confirmations == ["Also delete the intermediate files created by this workflow?"]
    assert not artifacts.book_path.exists()


def test_clear_skips_prompt_when_no_work_files_exist(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "novel.mp3"
    audio.write_bytes(b"audio")
    state = make_state(audio)
    confirm = Mock(return_value=True)
    monkeypatch.setattr(enhance_flow.ask, "ask_confirm", confirm)
    monkeypatch.setattr(state.prefs, "save", lambda: "")

    enhance_flow.clear_selection(state)

    confirm.assert_not_called()
