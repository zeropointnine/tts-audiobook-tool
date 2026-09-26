from __future__ import annotations

from tts_audiobook_tool.app_types import Hint

"""Business operations for the resumable enhance-audiobook submenu."""

import os
from pathlib import Path
import tempfile

from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.app_support import app_hint_util, hints
from tts_audiobook_tool.app_types.app_metadata import AppMetadata, AppMetadataTextSegment
from tts_audiobook_tool.constants import ABR_VERSION, COL_ACCENT, COL_DEFAULT, COL_DIM_ITALICS
from tts_audiobook_tool.constants_hints import HINT_ENHANCE_ORPHANS, HINT_STT_ENHANCE, HINT_STT_ENHANCE_CACHED
from tts_audiobook_tool.enhance import enhance_alignment, enhance_text
from tts_audiobook_tool.enhance.enhance_artifacts import (
    EnhanceArtifacts,
    count_misalignments,
    delete_temporary_files,
    load_output_timed_phrases,
    load_timed_phrases,
    load_transcription,
    make_enhance_state,
    save_book,
    save_timed_phrases,
    save_transcription,
)
from tts_audiobook_tool.enhance.unmatched_lines_app import UnmatchedLinesApp
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual.content_textual_app import ContentAppCompleted, run_content_textual_app
from tts_audiobook_tool.util import make_error_string, print_feedback, printt


SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".flac", ".mp4", ".m4a", ".m4b"}
COPY_AUDIO_SUFFIXES = {".m4a", ".m4b"}


def _validate_audio(path: Path) -> str:
    if not path.exists():
        return f"Audio file does not exist: {path}"
    if not path.is_file():
        return f"Audio path is not a regular file: {path}"
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        return f"Audio file suffix must be one of: {supported}"
    try:
        return SoundFileUtil.is_valid_sound_file(str(path))
    except Exception as exception:
        return f"Could not read audio file: {make_error_string(exception)}"


def _selected_artifacts(state: State) -> tuple[EnhanceArtifacts | None, str]:
    selected = state.prefs.enhance_audio_path
    if not selected:
        return None, "Enter an audiobook file path first."
    artifacts = EnhanceArtifacts.from_audio_path(selected)
    audio_error = _validate_audio(artifacts.audio_path)
    if audio_error:
        return None, audio_error
    return artifacts, ""


def _existing_output_path(artifacts: EnhanceArtifacts) -> Path | None:
    """
    Return the parallel enhanced audiobook for this audio, if any.

    The expected suffix depends on the book's text_source_kind, which is not
    knowable until source text is entered (the book artifact may be missing
    entirely), so both candidate names are checked.
    """
    for candidate in (artifacts.epub_output_path, artifacts.flat_output_path):
        if candidate.is_file():
            return candidate
    return None


def select_audio(state: State) -> None:
    current = state.prefs.enhance_audio_path
    current_parent = Path(current).parent if current else None
    if current_parent and current_parent.is_dir():
        initial_dir = str(current_parent)
    elif state.prefs.last_enhanced_dir and Path(state.prefs.last_enhanced_dir).is_dir():
        initial_dir = state.prefs.last_enhanced_dir
    else:
        initial_dir = ""
    selected = ask.ask_file_path(
        "Enter source audiobook file path: ",
        "Select audiobook file",
        filetypes=[
            ("Supported audio", "*.mp3 *.flac *.mp4 *.m4a *.m4b"),
            ("All files", "*.*"),
        ],
        initialdir=initial_dir,
    )
    if not selected:
        return

    path = Path(selected).resolve()
    error = _validate_audio(path)
    if error:
        ask.ask_error(error)
        return

    try:
        metadata = AppMetadata.load_from_file(str(path))
    except Exception as exception:
        ask.ask_error(f"Could not inspect audiobook metadata: {make_error_string(exception)}")
        return
    if metadata is not None and not ask.ask_confirm(
        "Audio file already has tts-audiobook-tool metadata. Continue anyway? "
    ):
        return

    # Existing sibling work files belong to a resumable run; Clear is the
    # explicit place to delete them, not selecting this audio.
    old_last_enhanced_dir = state.prefs.last_enhanced_dir
    state.prefs.enhance_audio_path = str(path)
    state.prefs.last_enhanced_dir = str(path.parent)
    save_error = state.prefs.save()
    if save_error:
        state.prefs.enhance_audio_path = current
        state.prefs.last_enhanced_dir = old_last_enhanced_dir
        ask.ask_error(f"Could not save audiobook selection: {save_error}")
        return

    existing = _existing_output_path(EnhanceArtifacts.from_audio_path(path))
    if existing is not None:
        existing = text_util.make_terminal_hyperlink(str(existing), is_file=True)
        hints.show_hint(
            Hint(
                "",
                f"An enhanced audiobook already exists at {existing}.\n",
                "You can review it from the menu, or redo the intermediate steps to replace it."
            ),
            and_prompt=True
        )

def select_text(state: State) -> None:
    artifacts, error = _selected_artifacts(state)
    if artifacts is None:
        ask.ask_error(error)
        return

    hints.show_hint_if_necessary(state.prefs, HINT_STT_ENHANCE)
    selected = ask.ask_file_path(
        "Enter text or EPUB file path: ",
        "Select text or EPUB file",
        filetypes=[
            ("Text and EPUB files", "*.txt *.epub"),
            ("Text files", "*.txt"),
            ("EPUB files", "*.epub"),
        ],
        initialdir=(
            state.prefs.last_text_dir
            if state.prefs.last_text_dir and Path(state.prefs.last_text_dir).is_dir()
            else ""
        ),
    )
    if not selected:
        return
    source_path = Path(selected)
    if not source_path.is_file():
        ask.ask_error(f"Source text file does not exist: {source_path}")
        return
    if source_path.suffix.lower() not in {".txt", ".epub"}:
        ask.ask_error("Source text file must have a .txt or .epub suffix.")
        return

    old_last_text_dir = state.prefs.last_text_dir
    state.prefs.last_text_dir = str(source_path.parent)
    prefs_error = state.prefs.save()
    if prefs_error:
        state.prefs.last_text_dir = old_last_text_dir
        ask.ask_error(f"Could not save the source-text directory: {prefs_error}")
        return

    if artifacts.book_path.exists() and not ask.ask_confirm(
        "Replace the existing imported source text? "
    ):
        return

    book = enhance_text.import_source_book(state, source_path)
    if book is None:
        return
    if isinstance(book, str):
        ask.ask_error(book)
        return

    save_error = save_book(artifacts, book)
    if save_error:
        ask.ask_error(save_error)
        return

    # Only the timed alignment depends on the source text; the audio-only
    # transcription can be reused. A completed final output deliberately
    # remains until Create is explicitly selected again.
    try:
        artifacts.timed_phrases_path.unlink(missing_ok=True)
    except OSError as exception:
        ask.ask_error(
            "Source text was saved, but the old timed-phrases cache could not be removed: "
            f"{make_error_string(exception)}"
        )
        return

    if book.text_source_kind == "plain_text":
        count = len(enhance_text.flatten_book(book).phrases)
        noun = "line" if count == 1 else "lines"
        print_feedback(f"Imported text ({count} {noun})")
    else:
        ask.ask_enter_to_continue()


def transcribe(state: State) -> None:
    artifacts, error = _selected_artifacts(state)
    if artifacts is None:
        ask.ask_error(error)
        return
    snapshot = make_enhance_state(state.prefs.enhance_audio_path)
    if snapshot.transcription_valid and not ask.ask_confirm(
        "Replace the existing transcription? "
    ):
        return
    from tts_audiobook_tool.enhance import enhance_menu

    # Transcription needs only the audio, so no book is required here. The
    # chain question needs the book (for its output suffix), and alignment
    # requires it anyway, so skip it when no source text has been entered yet.
    align_when_finished = False
    if snapshot.book is not None:
        suffix = enhance_menu.output_suffix(snapshot)
        align_when_finished = ask.ask_confirm(
            f"When transcription is finished, perform force-alignment step and create the \"{suffix}\" file? "
        )

    transcribing_line = (
        f"{COL_ACCENT}Transcribing audio "
        f"{COL_DIM_ITALICS}(This may take some time...){COL_DEFAULT}"
    )
    divider = "-" * len(text_util.strip_ansi_codes(transcribing_line))
    MenuUtil.print_heading(
        None,
        f"{COL_ACCENT}{divider}\n{transcribing_line}",
        dont_clear=True,
        non_menu=True,
    )
    try:
        words = enhance_alignment.transcribe_to_words(str(artifacts.audio_path), state.prefs)
    except Exception as exception:
        ask.ask_error(f"Transcription failed: {make_error_string(exception)}")
        return
    if words is None:
        printt()
        print_feedback("Interrupted")
        return

    save_error = save_transcription(artifacts, words)
    if save_error:
        ask.ask_error(save_error)
        return
    try:
        artifacts.timed_phrases_path.unlink(missing_ok=True)
    except OSError as exception:
        ask.ask_error(
            "Transcription was saved, but the old timed-phrases cache could not be removed: "
            f"{make_error_string(exception)}"
        )
        return

    printt("\a")
    print_feedback("Transcribing finished")
    hints.show_hint_if_necessary(state.prefs, HINT_STT_ENHANCE_CACHED)
    if align_when_finished:
        align_source_text(state, create_when_finished=True)


def align_source_text(state: State, *, create_when_finished: bool | None = None) -> None:
    artifacts, error = _selected_artifacts(state)
    if artifacts is None:
        ask.ask_error(error)
        return
    snapshot = make_enhance_state(state.prefs.enhance_audio_path)
    if snapshot.book is None:
        ask.ask_error(snapshot.book_error or "Source text required.")
        return
    words, transcription_error = load_transcription(artifacts)
    if words is None:
        ask.ask_error(transcription_error or "Transcription required.")
        return

    flattened = enhance_text.flatten_book(snapshot.book)
    if not flattened.phrases:
        ask.ask_error("The imported source book has no text segments.")
        return

    if create_when_finished is None:
        from tts_audiobook_tool.enhance import enhance_menu

        suffix = enhance_menu.output_suffix(snapshot)
        create_when_finished = ask.ask_confirm(
            f"Create the \"{suffix}\" file when alignment is finished? "
        )

    merging_line = f"{COL_ACCENT}Merging data...{COL_DEFAULT}"
    merging_divider = "-" * len(text_util.strip_ansi_codes(merging_line))
    MenuUtil.print_heading(
        None,
        f"{COL_ACCENT}{merging_divider}\n{merging_line}",
        dont_clear=True,
        non_menu=True,
    )
    try:
        timed_phrases, did_interrupt = enhance_text.align_book(snapshot.book, words)
    except Exception as exception:
        ask.ask_error(f"Alignment failed: {make_error_string(exception)}")
        return
    if did_interrupt:
        print_feedback("Interrupted")
        return
    if not timed_phrases:
        ask.ask_error("Alignment produced no timed text segments.")
        return

    timed_error = save_timed_phrases(artifacts, timed_phrases)
    if timed_error:
        ask.ask_error(timed_error)
        return

    if create_when_finished:
        create_output(state, overwrite=True)

    # The orphan review belongs to the alignment flow: it fires whether or not
    # the user opted into creating the output file, and after that step if so.
    orphan_count = count_misalignments(timed_phrases)
    if orphan_count and ask.ask_confirm("Review orphaned lines now? "):
        review_discontinuities(state)


def _make_staging_path(destination: Path) -> Path:
    descriptor, value = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    path = Path(value)
    path.unlink()
    return path


def _write_staged_output(
    state: State,
    artifacts: EnhanceArtifacts,
    destination: Path,
    metadata: AppMetadata,
) -> str:
    staging: Path | None = None
    try:
        staging = _make_staging_path(destination)
        source_suffix = artifacts.audio_path.suffix.lower()
        if source_suffix in COPY_AUDIO_SUFFIXES:
            error = AppMetadata.save_to_mp4(
                metadata,
                str(artifacts.audio_path),
                str(staging),
            )
        else:
            error = SoundFileUtil.transcode_to_aac_at(
                str(artifacts.audio_path),
                str(staging),
                state.prefs.aac_bitrate,
            )
            if not error:
                error = AppMetadata.save_to_mp4(metadata, str(staging))
        if error:
            return error
        try:
            readback = AppMetadata.load_from_file(str(staging))
        except Exception as exception:
            return f"Could not verify staged audiobook metadata: {make_error_string(exception)}"
        if readback is None:
            return "Could not verify staged audiobook metadata"
        os.replace(staging, destination)
        return ""
    except Exception as exception:
        return f"Could not create enhanced audiobook: {make_error_string(exception)}"
    finally:
        if staging is not None:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass


def create_output(state: State, overwrite: bool = False) -> None:
    artifacts, error = _selected_artifacts(state)
    if artifacts is None:
        ask.ask_error(error)
        return
    snapshot = make_enhance_state(state.prefs.enhance_audio_path)
    if snapshot.book is None:
        if snapshot.timed_phrases_valid:
            ask.ask_error(
                snapshot.book_error
                or (
                    "Source text required. The alignment was built from a book that is no "
                    "longer present; re-enter the source text (its alignment will be redone)."
                )
            )
        else:
            ask.ask_error(snapshot.book_error or "Source text required.")
        return
    timed_phrases, timed_error = load_timed_phrases(artifacts)
    if timed_phrases is None:
        ask.ask_error(timed_error or "Align source text with transcription before creating the output.")
        return
    if not timed_phrases:
        ask.ask_error("Alignment has no timed text segments. Align source text with transcription first.")
        return

    try:
        destination = artifacts.expected_output_path(snapshot.book)
    except ValueError as exception:
        ask.ask_error(str(exception))
        return
    if not overwrite and destination.exists() and not ask.ask_confirm(
        f"Replace the existing enhanced audiobook at {destination}? "
    ):
        return

    metadata_timed_phrases: list[AppMetadataTextSegment] = list(timed_phrases)
    metadata = AppMetadata(
        timed_phrases=metadata_timed_phrases,
        title=snapshot.book.title or artifacts.audio_path.stem,
        version=ABR_VERSION,
        bookmark_indices=[],
        raw_text="",
        has_break_audio=False,
        project_snapshot={},
        sections=enhance_text.make_app_metadata_sections(
            snapshot.book,
            len(timed_phrases),
        ),
    )
    saving_line = f"{COL_ACCENT}Creating audio file with added custom metadata{COL_DEFAULT}"
    divider = "-" * len(text_util.strip_ansi_codes(saving_line))
    printt(f"{COL_ACCENT}{divider}")
    printt(saving_line)
    printt()
    output_error = _write_staged_output(state, artifacts, destination, metadata)
    if output_error:
        ask.ask_error(output_error)
        return

    # print_feedback adds the nominal pause so the menu redraw doesn't
    # clear the "Saved" line before it can be read.
    print_feedback(
        f"\n{COL_ACCENT}Saved {COL_DEFAULT}{text_util.make_terminal_hyperlink(str(destination), is_file=True)}",
        no_preformat=True,
    )
    # These hints are followed by the menu redraw, which clears the screen;
    # the default 2-second hint animation is not long enough to read them.
    app_hint_util.show_player_hint(state.prefs, and_prompt=True)
    hints.show_hint_if_necessary(state.prefs, HINT_ENHANCE_ORPHANS, and_prompt=True)


def review_discontinuities(state: State) -> None:
    snapshot = make_enhance_state(state.prefs.enhance_audio_path)
    if snapshot.artifacts is None or snapshot.expected_output_path is None:
        ask.ask_error("Create an enhanced audiobook before reviewing discontinuities.")
        return
    if not snapshot.output_valid:
        ask.ask_error(snapshot.output_error or "The enhanced audiobook metadata is unreadable.")
        return

    timed_phrases, error = load_output_timed_phrases(snapshot.expected_output_path)
    if timed_phrases is None:
        ask.ask_error(error)
        return
    output_path = snapshot.expected_output_path
    audio_duration = AudioMetaUtil.get_audio_duration(str(output_path))
    result = run_content_textual_app(
        UnmatchedLinesApp(state.project, timed_phrases, output_path, audio_duration)
    )
    if not isinstance(result, ContentAppCompleted):
        ask.ask_error(result.message)


def clear_selection(state: State) -> None:
    current = state.prefs.enhance_audio_path
    if not current:
        return
    snapshot = make_enhance_state(current)
    errors: list[str] = []

    if (
        snapshot.artifacts is not None
        and any(path.exists() for path in snapshot.artifacts.temporary_paths)
        and ask.ask_confirm("Also delete the intermediate files created by this workflow?")
    ):
        errors.extend(delete_temporary_files(snapshot.artifacts))

    state.prefs.enhance_audio_path = ""
    save_error = state.prefs.save()
    if save_error:
        state.prefs.enhance_audio_path = current
        errors.append(f"Could not save the cleared audiobook selection: {save_error}")

    if errors:
        errors.append(
            "Reselecting the source audiobook makes any retained sibling work files discoverable again."
        )
        ask.ask_error("\n".join(errors))
