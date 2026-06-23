import os
from pathlib import Path
import pickle
import time
from dataclasses import dataclass, field
from typing import Any
from tts_audiobook_tool.app_support import app_hashing, app_paths, hints, app_hint_util
from tts_audiobook_tool.app_types.app_metadata import AppMetadata, AppMetadataSection
from tts_audiobook_tool.app_types import SttVariant, Word
from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.enhance import enhance_alignment
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.epub_menu_util import EpubMenuUtil
from tts_audiobook_tool.menus.menu_util import MenuUtil

from tts_audiobook_tool.util import *


@dataclass
class EnhanceSourceText:
    raw_text: str
    phrases: list[Phrase]
    source_kind: str = "text"
    title: str = ""
    section_ranges: list[tuple[int, int]] = field(default_factory=list)
    section_titles: list[str] = field(default_factory=list)


def ask_and_make(state: State) -> None:

    MenuUtil.print_screen_heading(state, "Enhance existing audiobook", "Enhance audiobook")

    hints.show_hint_if_necessary(state.prefs, HINT_STT_ENHANCE)

    # [1] Ask text/EPUB file
    if DEV and False:
        inp = r"exc.txt"
    else:
        inp = ask.ask_file_path(
            "Step 1/2 - Enter text or EPUB file path: ",
            "Step 1/2: Select text or EPUB file",
            filetypes=[("Text and EPUB files", "*.txt *.epub"), ("Text files", "*.txt"), ("EPUB files", "*.epub"), ("All files", "*.*")],
            initialdir=state.prefs.last_text_dir if state.prefs.last_text_dir and os.path.exists(state.prefs.last_text_dir) else "",
        )
    if not inp:
        return
    if not os.path.exists(inp):
        ask.ask_enter_to_continue(f"File doesn't exist.")
        return
    source_text_path = Path(inp)
    state.prefs.last_text_dir = str(source_text_path.parent)

    source_text = load_source_text_for_enhance(state, str(source_text_path))
    if source_text is None:
        return

    time.sleep(1)

    # [2] Ask audio file
    if DEV and False:
        inp = r"exc.flac"
    else:
        inp = ask.ask_file_path("Step 2/2 - Enter audiobook file path: ", "Step 2/2: Select audiobook file")

    if not inp:
        return
    if not os.path.exists(inp):
        printt(f"File doesn't exist.")
        return
    source_audio_path = inp

    # Optional transcode step
    if Path(source_audio_path).suffix == ".mp3":
        hints.show_hint_if_necessary(state.prefs, HINT_MULTIPLE_MP3S)
        b = ask.ask_confirm("MP3 file must first be transcoded to AAC. Do this now? ")
        if not b:
            return
        path, err = SoundFileUtil.transcode_to_aac(source_audio_path)
        if err:
            ask.ask_error(err)
            return
        source_audio_path = path

    # Make normalized, 'idempotent' path
    source_audio_path = str(Path(source_audio_path).resolve().as_posix())

    types = [".flac", ".mp4", ".m4a", ".m4b"]
    if not Path(source_audio_path).suffix in types:
        printt("File suffix must be one of the following: {types}")
        return

    # Check if already has meta
    meta = AppMetadata.load_from_file(source_audio_path)
    if meta is not None:
        if DEV and False:
            b = True
        else:
            b = ask.ask_confirm("Audio file already has tts-audiobook-tool metadata. Continue anyway? ")
        if not b:
            return

    # [3] Calc hash
    source_audio_hash, err = app_hashing.calc_hash_file(
        source_audio_path,
        print_progress_text="Calculating audio file hash:"
    )
    if err:
        ask.ask_error(err)
        return

    # [4] Check if already has transcription pickle file
    transcription_pickle_path = _make_transcription_pickle_file_path(source_audio_hash)
    if not os.path.exists(transcription_pickle_path):
        transcription_pickle_path = ""
    else:
        if DEV and False:
            b = True
        else:
            b = ask.ask_confirm("You've previously transcribed this audio file. Use saved transcription data? ")
        if not b:
            transcription_pickle_path = ""

    # TODO: new [5] Ask for language code hint

    # [5] Start
    make(
        state.prefs,
        source_text,
        source_audio_path=source_audio_path,
        source_audio_hash=source_audio_hash,
        source_pickle_path=transcription_pickle_path
    )


def make(
        prefs: Prefs,
        raw_text: str | EnhanceSourceText,
        source_audio_path: str,
        source_audio_hash: str,
        source_pickle_path: str=""
) -> bool:
    """
    Optional source_pickle_path is the already-transcribed data from source_audio_path,

    Returns True for success
    """

    # [1] Make segmented source text

    source_text = normalize_enhance_source_text(raw_text)
    if not source_text.raw_text:
        ask.ask_enter_to_continue("File has no content.")
        return False
    if not source_text.phrases:
        ask.ask_enter_to_continue("Source text produced no segments.")
        return False

    # [2] Transcribe audio file (or load pickle file)

    if source_pickle_path:

        try:
            with open(source_pickle_path, "rb") as file:
                words = pickle.load(file)
        except Exception as e:
            ask.ask_error(make_error_string(e))
            return False

    else:

        MenuUtil.print_heading(None, f"Transcribing audio... {COL_DIM}(This may take some time)", dont_clear=True, non_menu=True)
        printt()

        # Always use best whisper model
        Stt.set_variant(SttVariant.LARGE_V3)

        # Warm up
        _ = Stt.get_whisper()

        words = enhance_alignment.transcribe_to_words(str(source_audio_path))
        if words is None: # interrupted
            printt("")
            print_feedback("Interrupted")
            return False

        printt("\a")

        # Restore variant / clean up model if necessary
        Stt.set_variant(prefs.stt_variant)

        # Save transcription data to pickle
        pickle_path = _make_transcription_pickle_file_path(source_audio_hash)
        try:
            with open(pickle_path, "wb") as file:
                pickle.dump(words, file)
        except:
            pass # eat

    # [3] "Merge" source text and transcribed text data

    MenuUtil.print_heading(None, "Merging data...", dont_clear=True, non_menu=True)

    timed_phrases, did_interrupt = make_section_aware_timed_phrases(source_text, words)

    if did_interrupt:
        print_feedback("Interrupted")
        return False

    sections = make_app_metadata_sections(source_text, len(timed_phrases))

    # [4] Save "abr" audio file

    dest_name = Path(source_audio_path).stem + ".abr" + Path(source_audio_path).suffix # eg, "teh_hobbit.abr.m4b"
    dest_path = str( Path(source_audio_path).with_name(dest_name) )
    dest_path = make_unique_file_path(dest_path)
    printt(f"\nSaving audio file with added custom metadata")
    printt()

    meta = AppMetadata(
        timed_phrases=timed_phrases,
        title=source_text.title or Path(source_audio_path).stem,
        version=ABR_VERSION,
        bookmark_indices=[],
        raw_text=source_text.raw_text,
        has_break_audio=False,
        project_snapshot={},
        sections=sections
    )
    if dest_path.lower().endswith(".flac"):
        save_error = AppMetadata.save_to_flac(meta, str(source_audio_path), str(dest_path))
    else:
        save_error = AppMetadata.save_to_mp4(meta, str(source_audio_path), str(dest_path))
    if save_error:
        printt(f"{COL_ERROR}Error: {save_error}")
        printt()
    else:
        printt(f"{COL_ACCENT}Saved {dest_path}")
        printt()

    hints.show_hint_if_necessary(prefs, HINT_STT_ENHANCE_CACHED)

    # [4b] Review "discontinuity info"
    b = ask.ask_confirm("View discontinuity info summary? ")
    if b:
        print_discontinuity_info(timed_phrases)
        ask.ask_enter_to_continue()

    if not save_error:
        app_hint_util.show_player_hint(prefs)

    return bool(save_error)


def load_source_text_for_enhance(state: State | Any, source_text_path: str) -> EnhanceSourceText | None:
    suffix = Path(source_text_path).suffix.lower()
    if suffix == ".epub":
        epub_import_result = EpubMenuUtil.import_epub(
            epub_path=source_text_path,
            max_words=state.project.max_words,
            segmentation_strategy=state.project.segmentation_strategy,
            language_code=state.project.language_code,
        )
        if epub_import_result is None:
            return None
        if not epub_import_result.raw_text.strip():
            ask.ask_enter_to_continue("EPUB import produced no text.")
            return None
        if not epub_import_result.phrase_groups:
            ask.ask_enter_to_continue("EPUB import produced no text segments.")
            return None
        EpubMenuUtil.print_import_info(epub_import_result)
        phrases = PhraseGroup.flatten_groups(epub_import_result.phrase_groups)
        section_ranges = make_section_ranges_from_group_starts(
            epub_import_result.phrase_groups,
            epub_import_result.section_start_indices,
        )
        return EnhanceSourceText(
            raw_text=epub_import_result.raw_text,
            phrases=phrases,
            source_kind="epub",
            title=epub_import_result.book_title,
            section_ranges=section_ranges,
            section_titles=[chapter.title for chapter in epub_import_result.chapters],
        )

    try:
        with open(source_text_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
    except Exception as e:
        ask.ask_error(f"Error: {e}")
        return None

    if not raw_text:
        ask.ask_enter_to_continue("File has no content.")
        return None

    return normalize_enhance_source_text(raw_text)


def normalize_enhance_source_text(raw_text: str | EnhanceSourceText) -> EnhanceSourceText:
    if isinstance(raw_text, EnhanceSourceText):
        return raw_text

    printt("Segmenting source text...")
    printt()
    groups = PhraseGrouper.text_to_groups(raw_text, max_words=MAX_WORDS_PER_SEGMENT_STT)
    phrases = PhraseGroup.flatten_groups(groups)
    return EnhanceSourceText(raw_text=raw_text, phrases=phrases)


def make_section_ranges_from_group_starts(
        phrase_groups: list[PhraseGroup],
        section_start_indices: list[int],
) -> list[tuple[int, int]]:
    starts = [0]
    starts.extend(index for index in section_start_indices if 0 < index < len(phrase_groups))
    starts = sorted(set(starts))

    group_phrase_starts: list[int] = []
    phrase_index = 0
    for group in phrase_groups:
        group_phrase_starts.append(phrase_index)
        phrase_index += len(group.phrases)
    group_phrase_starts.append(phrase_index)

    ranges: list[tuple[int, int]] = []
    for section_index, start_group_index in enumerate(starts):
        end_group_index = starts[section_index + 1] if section_index + 1 < len(starts) else len(phrase_groups)
        start_phrase_index = group_phrase_starts[start_group_index]
        end_phrase_index = group_phrase_starts[end_group_index]
        if end_phrase_index > start_phrase_index:
            ranges.append((start_phrase_index, end_phrase_index))
    return ranges


def make_section_aware_timed_phrases(
        source_text: EnhanceSourceText,
        words: list[Word],
) -> tuple[list[TimedPhrase], bool]:
    if not source_text.section_ranges:
        return enhance_alignment.make_timed_phrases(source_text.phrases, words)

    timed_phrases: list[TimedPhrase] = []
    alignment_state = enhance_alignment.AlignmentState()
    line_offset = 0
    for section_index, (start, end) in enumerate(source_text.section_ranges):
        section_phrases = source_text.phrases[start:end]
        if not section_phrases:
            continue
        if len(source_text.section_ranges) > 1:
            title = source_text.section_titles[section_index] if section_index < len(source_text.section_titles) else ""
            title_suffix = f": {title}" if title else ""
            printt(f"{COL_ACCENT}Aligning EPUB section {section_index + 1}/{len(source_text.section_ranges)}{title_suffix}{COL_DEFAULT}")
            printt()
        section_timed_phrases, alignment_state, did_interrupt = enhance_alignment.align_phrases_with_state(
            section_phrases,
            words,
            alignment_state,
            line_offset=line_offset,
        )
        if did_interrupt:
            return [], True
        timed_phrases.extend(section_timed_phrases)
        line_offset += len(section_phrases)

    return timed_phrases, False


def make_app_metadata_sections(source_text: EnhanceSourceText, text_segment_count: int) -> list[AppMetadataSection]:
    sections: list[AppMetadataSection] = []
    for section_index, (start, end) in enumerate(source_text.section_ranges):
        if start >= text_segment_count:
            continue
        section_end = min(end, text_segment_count)
        if section_end <= start:
            continue
        title = source_text.section_titles[section_index] if section_index < len(source_text.section_titles) else ""
        sections.append(AppMetadataSection(
            title=title,
            start_index=start,
            end_index=section_end,
        ))
    return sections


def print_discontinuity_info(timed_text_segments: list[TimedPhrase]):

    MenuUtil.print_heading(None, "Unmatched text segments:", dont_clear=True, non_menu=True)
    printt()

    discon_ranges = TimedPhrase.get_discontinuities(timed_text_segments)
    if not discon_ranges:
        printt("No items found")
        printt()
        return

    for start, end in discon_ranges:

        num_consecutive = end - start + 1

        if num_consecutive == 1:
            printt(f"Line {start}")
            text = timed_text_segments[start].text.strip()
            printt(f"    {COL_DIM}Line:{COL_DEFAULT} {ellipsize(text, 50)}")
            printt()
        else:
            printt(f"Lines {start} to {end} ({COL_ERROR}{num_consecutive} lines{COL_DEFAULT})")
            first_text = timed_text_segments[start].text.strip()
            last_text = timed_text_segments[end].text.strip()
            printt(f"    {COL_DIM}First line:{COL_DEFAULT} {ellipsize(first_text, 50)}")
            printt(f"    {COL_DIM}Last line:{COL_DEFAULT} {ellipsize(last_text, 50)}")
            printt()


def _make_transcription_pickle_file_path(hash: str) -> str:
    file_name = f"transcription {hash}.pkl"
    return os.path.join(app_paths.get_app_user_dir(), file_name)
