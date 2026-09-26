from __future__ import annotations

"""Deterministic, resumable artifacts for enhancing an existing audiobook.

Completion intentionally means "present and basically readable", not "newer than
all of its inputs". Replacing an upstream artifact therefore does not delete a
finished audiobook; choosing Create again is the explicit rebuild operation.
This keeps persistence independent from menu presentation and leaves room for a
future fingerprint/freshness policy.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Callable, Iterable, TypeVar

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Book, Word
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.app_types.book_serialization import (
    BOOK_FORMAT,
    book_from_project_text_json_dict,
    book_to_project_text_json_dict,
)
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import make_error_string


T = TypeVar("T")


@dataclass(frozen=True)
class EnhanceArtifacts:
    audio_path: Path
    book_path: Path
    transcription_path: Path
    timed_phrases_path: Path
    flat_output_path: Path
    epub_output_path: Path

    @classmethod
    def from_audio_path(cls, audio_path: str | os.PathLike[str]) -> "EnhanceArtifacts":
        audio = Path(audio_path)
        stem = audio.stem
        return cls(
            audio_path=audio,
            book_path=audio.with_name(f"{stem}.abr.json"),
            transcription_path=audio.with_name(f"{stem}.transcription.bin"),
            timed_phrases_path=audio.with_name(f"{stem}.timed_phrases.bin"),
            flat_output_path=audio.with_name(f"{stem}.abr.m4a"),
            epub_output_path=audio.with_name(f"{stem}.abr.m4b"),
        )

    @property
    def temporary_paths(self) -> tuple[Path, Path, Path]:
        return self.book_path, self.transcription_path, self.timed_phrases_path

    def expected_output_path(self, book: Book) -> Path:
        if book.text_source_kind == "plain_text":
            return self.flat_output_path
        if book.text_source_kind == "epub":
            return self.epub_output_path
        raise ValueError(
            f"Unsupported enhance text source kind: {book.text_source_kind or '(missing)'}"
        )


@dataclass(frozen=True)
class EnhanceState:
    artifacts: EnhanceArtifacts | None
    has_audio_path: bool
    audio_exists: bool
    book: Book | None
    book_error: str
    transcription_exists: bool
    transcription_valid: bool
    transcription_error: str
    timed_phrases_exists: bool
    timed_phrases_valid: bool
    timed_phrases_error: str
    expected_output_path: Path | None
    output_exists: bool
    output_valid: bool
    output_error: str
    output_phrase_count: int
    misalignment_count: int


def save_book(artifacts: EnhanceArtifacts, book: Book) -> str:
    return JsonSaveUtil.save(
        JsonArtifactType.ENHANCE_BOOK,
        artifacts.book_path,
        lambda: book_to_project_text_json_dict(book),
    )


def load_book(artifacts: EnhanceArtifacts) -> tuple[Book | None, str]:
    if not artifacts.book_path.is_file():
        return None, "Source book file is missing"
    try:
        with artifacts.book_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exception:
        return None, f"Error loading source book: {make_error_string(exception)}"
    if not isinstance(payload, dict) or payload.get("format") != BOOK_FORMAT:
        return None, f"Source book must use the {BOOK_FORMAT} format"
    try:
        result = book_from_project_text_json_dict(payload)
    except Exception as exception:
        return None, f"Error parsing source book: {make_error_string(exception)}"
    if isinstance(result, str):
        return None, f"Error parsing source book: {result}"
    if result.text_source_kind not in ("plain_text", "epub"):
        return None, (
            "Source book has unsupported text_source_kind: "
            f"{result.text_source_kind or '(missing)'}"
        )
    if result.audio_source_kind != "pre_existing":
        return None, (
            "Source book has invalid audio_source_kind: "
            f"{result.audio_source_kind or '(missing)'}"
        )
    return result, ""


def atomic_pickle_save(path: str | os.PathLike[str], payload: Any) -> str:
    destination = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            pickle.dump(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return ""
    except Exception as exception:
        return f"Error saving {destination.name}: {make_error_string(exception)}"
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _load_pickle_list(
    path: Path,
    item_validator: Callable[[Any], bool],
    artifact_name: str,
) -> tuple[list[Any] | None, str]:
    if not path.is_file():
        return None, f"{artifact_name} file is missing"
    try:
        with path.open("rb") as file:
            value = pickle.load(file)
    except Exception as exception:
        return None, f"Error loading {artifact_name}: {make_error_string(exception)}"
    if not isinstance(value, list):
        return None, f"Invalid {artifact_name}: expected a list"
    for index, item in enumerate(value):
        if not item_validator(item):
            return None, f"Invalid {artifact_name} item at index {index}"
    return value, ""


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def is_word(value: Any) -> bool:
    return (
        _is_number(getattr(value, "start", None))
        and _is_number(getattr(value, "end", None))
        and isinstance(getattr(value, "word", None), str)
        and _is_number(getattr(value, "probability", None))
    )


def is_timed_phrase(value: Any) -> bool:
    return (
        isinstance(getattr(value, "text", None), str)
        and _is_number(getattr(value, "time_start", None))
        and _is_number(getattr(value, "time_end", None))
    )


def load_transcription(artifacts: EnhanceArtifacts) -> tuple[list[Word] | None, str]:
    value, error = _load_pickle_list(
        artifacts.transcription_path,
        is_word,
        "transcription",
    )
    return value, error  # type: ignore[return-value]


def save_transcription(artifacts: EnhanceArtifacts, words: list[Word]) -> str:
    if not isinstance(words, list) or not all(is_word(word) for word in words):
        return "Invalid transcription: expected a list of words"
    return atomic_pickle_save(artifacts.transcription_path, words)


def load_timed_phrases(
    artifacts: EnhanceArtifacts,
) -> tuple[list[TimedPhrase] | None, str]:
    value, error = _load_pickle_list(
        artifacts.timed_phrases_path,
        is_timed_phrase,
        "timed phrases",
    )
    return value, error  # type: ignore[return-value]


def save_timed_phrases(
    artifacts: EnhanceArtifacts,
    timed_phrases: list[TimedPhrase],
) -> str:
    if not isinstance(timed_phrases, list) or not all(
        is_timed_phrase(item) for item in timed_phrases
    ):
        return "Invalid timed phrases: expected a list of TimedPhrase records"
    return atomic_pickle_save(artifacts.timed_phrases_path, timed_phrases)


def flatten_metadata_timed_phrases(metadata: AppMetadata) -> list[TimedPhrase]:
    result: list[TimedPhrase] = []
    for item in metadata.timed_phrases:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def count_misalignments(timed_phrases: Iterable[TimedPhrase]) -> int:
    return sum(
        1
        for item in timed_phrases
        if item.time_start == 0.0 and item.time_end == 0.0
    )


def load_output_timed_phrases(path: Path) -> tuple[list[TimedPhrase] | None, str]:
    if not path.is_file():
        return None, "Enhanced output file is missing"
    try:
        metadata = AppMetadata.load_from_file(str(path))
    except Exception as exception:
        return None, f"Error reading enhanced output metadata: {make_error_string(exception)}"
    if metadata is None:
        return None, "Enhanced output has no readable app metadata"
    phrases = flatten_metadata_timed_phrases(metadata)
    if not phrases or not all(is_timed_phrase(item) for item in phrases):
        return None, "Enhanced output has invalid or empty timed metadata"
    return phrases, ""


def delete_temporary_files(artifacts: EnhanceArtifacts) -> list[str]:
    """Delete only the three documented work files and aggregate failures."""
    errors: list[str] = []
    for path in artifacts.temporary_paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exception:
            errors.append(f"{path}: {make_error_string(exception)}")
    return errors


def make_enhance_state(enhance_audio_path: str) -> EnhanceState:
    if not enhance_audio_path:
        return EnhanceState(
            artifacts=None,
            has_audio_path=False,
            audio_exists=False,
            book=None,
            book_error="",
            transcription_exists=False,
            transcription_valid=False,
            transcription_error="",
            timed_phrases_exists=False,
            timed_phrases_valid=False,
            timed_phrases_error="",
            expected_output_path=None,
            output_exists=False,
            output_valid=False,
            output_error="",
            output_phrase_count=0,
            misalignment_count=0,
        )

    artifacts = EnhanceArtifacts.from_audio_path(enhance_audio_path)
    audio_exists = artifacts.audio_path.is_file()
    book_exists = artifacts.book_path.is_file()
    book, book_error = load_book(artifacts) if book_exists else (None, "")

    transcription_exists = artifacts.transcription_path.is_file()
    transcription, transcription_error = (
        load_transcription(artifacts) if transcription_exists else (None, "")
    )
    timed_exists = artifacts.timed_phrases_path.is_file()
    timed_phrases, timed_error = (
        load_timed_phrases(artifacts) if timed_exists else (None, "")
    )

    expected_output_path: Path | None = None
    output_exists = False
    output_valid = False
    output_error = ""
    output_phrases: list[TimedPhrase] | None = None
    if book is not None:
        expected_output_path = artifacts.expected_output_path(book)
        output_exists = expected_output_path.is_file()
        if output_exists:
            output_phrases, output_error = load_output_timed_phrases(expected_output_path)
            output_valid = output_phrases is not None

    # Review plays the finished output, which may predate the latest alignment.
    # Its count and total must describe the phrases embedded in that output.
    return EnhanceState(
        artifacts=artifacts,
        has_audio_path=True,
        audio_exists=audio_exists,
        book=book,
        book_error=book_error,
        transcription_exists=transcription_exists,
        transcription_valid=transcription is not None,
        transcription_error=transcription_error,
        timed_phrases_exists=timed_exists,
        timed_phrases_valid=timed_phrases is not None,
        timed_phrases_error=timed_error,
        expected_output_path=expected_output_path,
        output_exists=output_exists,
        output_valid=output_valid,
        output_error=output_error,
        output_phrase_count=len(output_phrases or []),
        misalignment_count=count_misalignments(output_phrases or []),
    )


# Memoized snapshots for menu rendering. Deriving a state unpickles the whole
# transcription and decodes the output file's ABR tag, which is too expensive
# to pay on every redraw, so the last snapshot per audio path is kept and
# validated against a stat signature of every file it was derived from.
# All writers in this module use atomic replace, which always bumps
# st_mtime_ns, so the signature cannot miss a mutation in practice.
# The flows deliberately call the uncached make_enhance_state() after they
# mutate files, so this cache affects presentation only.
_state_cache: dict[str, tuple[tuple[tuple[int, int], ...], EnhanceState]] = {}


def _safe_stat_signature(paths: tuple[Path, ...]) -> tuple[tuple[int, int], ...]:
    signature: list[tuple[int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            signature.append((-1, -1))
        else:
            signature.append((stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _state_signature(state: EnhanceState) -> tuple[tuple[int, int], ...]:
    assert state.artifacts is not None
    paths = [state.artifacts.audio_path, *state.artifacts.temporary_paths]
    if state.expected_output_path is not None:
        paths.append(state.expected_output_path)
    return _safe_stat_signature(tuple(paths))


def invalidate_state_cache(enhance_audio_path: str = "") -> None:
    """Drop one cached snapshot, or the whole cache with an empty path."""
    if enhance_audio_path:
        _state_cache.pop(enhance_audio_path, None)
    else:
        _state_cache.clear()


def make_enhance_state_cached(enhance_audio_path: str) -> EnhanceState:
    """
    Return make_enhance_state(), reusing the previous snapshot when the stat
    signature of its source files is unchanged.

    The expected output path is itself derived from the book file, which is
    part of the signature, so a changed book invalidates via the book's stat
    rather than via an output path that no longer matches it.
    """
    cached = _state_cache.get(enhance_audio_path)
    if cached is not None:
        signature, state = cached
        if _state_signature(state) == signature:
            return state
    state = make_enhance_state(enhance_audio_path)
    if state.artifacts is not None:
        _state_cache[enhance_audio_path] = (_state_signature(state), state)
    else:
        _state_cache.pop(enhance_audio_path, None)
    return state
