from __future__ import annotations

"""
Enhance-flow alignment helpers.

This module supports the "enhance existing audiobook" flow by:
- transcribing long audio into word timestamps using overlapping chunks,
- stitching chunk transcripts into a single word stream,
- aligning source phrases to transcript words to build TimedPhrases.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generator, List, NamedTuple
import logging
import subprocess
import time
import ffmpeg
import numpy as np
import difflib
from tts_audiobook_tool.app_types import ConcreteWord, SttVariant, Word
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.model_worker import ModelWorker, STT_TRANSCRIPTION_TIMEOUT_ERROR
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.app_types.phrase import Phrase
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import *
from tts_audiobook_tool.transcriber import Transcriber


@dataclass
class AlignmentState:
    cursor: int = 0
    max_skip_words: int = 45
    orphan_count: int = 0


def make_timed_phrases(
    phrases: List[Phrase],
    transcribed_words: List[Word],
    print_info: bool=True
) -> tuple[ List[TimedPhrase], bool ]:
    """
    Forced-alignment algorithm
    Takes in source text list of TextSegments and list of transcribed Words
    to create list of TimedPhrases.

    Returns list and if did interrupt
    """

    if DEBUG:
        print(Ansi.CLEAR_SCREEN_AND_SCROLLBACK)
        print("\nsource text:\n")
        for i, item in enumerate(phrases):
            print(i, item.text.strip())
        s = ""
        for item in transcribed_words:
            s += item.word + " "
        s = normalize_text(s)
        print(f"\ntranscribed text:\n{s}\n")

    Interrupts().set("thinking")
    debug_start_time = time.time()

    timed_phrases, _, did_interrupt = align_phrases_with_state(
        phrases,
        transcribed_words,
        AlignmentState(),
        print_info=print_info,
    )

    if did_interrupt:
        Interrupts().clear()
        return [], True

    if DEBUG:
        elapsed_sec = (time.time() - debug_start_time)
        printt(f"\nElapsed: {elapsed_sec}\n")

        for i, item in enumerate(timed_phrases):
            printt(f"{i}  {item}")
        printt()

    Interrupts().clear()
    return timed_phrases, False


def align_phrases_with_state(
    phrases: List[Phrase],
    transcribed_words: List[Word],
    state: AlignmentState | None=None,
    print_info: bool=True,
    line_offset: int=0,
    total_lines: int=0,
) -> tuple[List[TimedPhrase], AlignmentState, bool]:
    """
    Aligns phrases against transcribed words while preserving a caller-owned transcript cursor.

    This is used by EPUB enhancement to align each section independently without resetting
    transcript progress at section boundaries.
    """

    result: List[TimedPhrase] = []
    state = state or AlignmentState()
    progress_total = total_lines or len(phrases)

    def show_progress(segment_index: int) -> None:
        if print_info and not PRINT_VERBOSE:
            line_number = line_offset + segment_index + 1
            print(
                f"{Ansi.LINE_HOME}{COL_DIM_ITALICS}Aligning {line_number}/{progress_total} "
                f"(orphans: {state.orphan_count}){Ansi.RESET}{Ansi.ERASE_REST_OF_LINE}",
                end="",
                flush=True,
            )

    def record(timed_phrase: TimedPhrase) -> None:
        result.append(timed_phrase)
        if timed_phrase.time_start == 0.0 and timed_phrase.time_end == 0.0:
            state.orphan_count += 1

    if not phrases:
        return [], state, False
    if not transcribed_words:
        for segment_index, phrase in enumerate(phrases):
            record(TimedPhrase(phrase.text, 0.0, 0.0))
            show_progress(segment_index)
        return result, state, False

    current_skip_span = 0

    for segment_index, segment in enumerate(phrases):

        if Interrupts().did_interrupt:
            return [], state, True

        show_progress(segment_index)
        orphans_before_segment = state.orphan_count

        text_normed = normalize_text(segment.text)
        num_words = len(text_normed.split())

        if not text_normed:
            record(TimedPhrase(segment.text, 0.0, 0.0))
            show_progress(segment_index)
            continue

        best_match: MatchInfo | None = None

        trans_index_start_min = state.cursor
        trans_index_start_max = state.cursor + state.max_skip_words
        trans_index_start_max = min(trans_index_start_max, len(transcribed_words))

        for trans_index_start in range(trans_index_start_min, trans_index_start_max):

            phrase_length_min = int(num_words * 0.75) or 1
            extra = int(num_words * 0.25)
            extra = max(extra, 2)
            phrase_length_max = num_words + extra

            trans_index_end_min = trans_index_start + phrase_length_min
            trans_index_end_max = trans_index_start + phrase_length_max
            trans_index_end_max = min(trans_index_end_max, len(transcribed_words))

            for trans_index_end in range(trans_index_end_min, trans_index_end_max + 1):

                trans_words = transcribed_words[trans_index_start : trans_index_end]
                if not trans_words:
                    continue
                trans_text_normed = " ".join(
                    normalize_text(item.word) for item in trans_words
                )
                if not trans_text_normed:
                    continue

                modded_segment_text_normed = text_normed
                modded_trans_text_normed = trans_text_normed

                if segment_index > 0 and state.cursor > 0 and trans_index_start > 0:
                    previous_segment = phrases[segment_index - 1]
                    previous_segment_word = normalize_text(previous_segment.text).split(" ")[-1]
                    previous_segment_word = normalize_text(previous_segment_word)

                    previous_trans_word = transcribed_words[state.cursor - 1].word
                    previous_trans_word = normalize_text(previous_trans_word)

                    if previous_segment_word and previous_trans_word:
                        modded_segment_text_normed = previous_segment_word + " " + text_normed
                        modded_trans_text_normed = previous_trans_word + " " + trans_text_normed

                matcher = difflib.SequenceMatcher(None, modded_segment_text_normed, modded_trans_text_normed)
                score = matcher.ratio()

                if not best_match or score > best_match.score:
                    best_match = MatchInfo(
                        trans_index_start=trans_index_start,
                        trans_index_end=trans_index_end,
                        trans_text=trans_text_normed,
                        score=score
                    )
                    current_skip_span = trans_index_start - trans_index_start_min

        def print_result(is_success: bool):
            printt(f"line {line_offset + segment_index + 1}")
            printt(f"{COL_DIM}    source text: {truncate_pretty(text_normed, 50)}")
            if best_match:
                printt(f"{COL_DIM}    transcribed: {truncate_pretty(best_match.trans_text, 50)}")
            s = f"{COL_DIM}    score: {best_match.score if best_match else 0}"
            if is_success:
                s += f" - {COL_OK}OK"
            else:
                s += f" - {COL_ERROR}Failed"
            printt(s)
            printt()

        if best_match and best_match.score >= MIN_MATCH_RATIO:
            start_time = transcribed_words[best_match.trans_index_start].start
            if best_match.trans_index_end < len(transcribed_words):
                end_time = transcribed_words[best_match.trans_index_end].start
            else:
                end_time = transcribed_words[best_match.trans_index_end - 1].end
            record(
                TimedPhrase(
                    segment.text,
                    start_time,
                    end_time
            ))

            if print_info and PRINT_VERBOSE:
                print_result(True)

            state.cursor = best_match.trans_index_end
            state.max_skip_words = MAX_SKIP_WORDS_BASE

            if DEBUG and False:
                if current_skip_span:
                    print(f"had to skip {current_skip_span} words")
                print(f"cursor is now: [{state.cursor+1}] {make_words_string(transcribed_words, state.cursor)}")
                print()

        else:
            record(
                TimedPhrase(segment.text, 0.0, 0.0)
            )

            was_max_skip = state.max_skip_words
            state.max_skip_words += len( segment.text.split(" ") ) * 2
            state.max_skip_words = min(state.max_skip_words, MAX_SKIP_WORDS_LIMIT)

            if print_info and PRINT_VERBOSE:
                print_result(False)

            if DEBUG and PRINT_VERBOSE:
                print(f"scanned transcript in this range: {make_words_string(transcribed_words, state.cursor, was_max_skip)}")
                print(f"cursor stays at: [{state.cursor+1}] ")
                print(f"max_skip_words has increased to: {state.max_skip_words}")
                print()

        if state.orphan_count != orphans_before_segment:
            show_progress(segment_index)

        if state.cursor >= len(transcribed_words) - 1:
            for remaining_segment_index in range(segment_index + 1, len(phrases)):
                remaining_seg = phrases[remaining_segment_index]
                record(TimedPhrase(
                    remaining_seg.text, 0.0, 0.0
                ))
            if segment_index + 1 < len(phrases):
                show_progress(len(phrases) - 1)
            break

    return result, state, False


def transcribe_to_words(path: str, prefs: Prefs, language_code: str) -> list[Word] | None:
    """
    Creates a list of Word instances by transcribing the audio at the given file path
    Returns None if interrupted
    """
    list_of_lists = _transcribe_stream_with_overlap(path, prefs, language_code)
    if list_of_lists is None:
        return None
    words_list = _stitch_transcripts(list_of_lists)
    return words_list


def _transcribe_stream_with_overlap(
    path: str, prefs: Prefs, language_code: str
) -> list[list[Word]] | None:
    CHUNK_DURATION = 30
    OVERLAP_DURATION = 5

    stt_variant = getattr(prefs, "stt_variant", None) or SttVariant.LARGE_V3
    language = language_code or None
    if stt_variant == SttVariant.DISABLED:
        raise ValueError("Speech-to-text is disabled in preferences; choose a Whisper model to transcribe.")

    list_of_lists: list[list[Word]] = []
    time_offset = 0.0

    duration_str = ""
    value = AudioMetaUtil.get_audio_duration(path)
    if value:
        duration_str = duration_string(value)

    log = logging.getLogger("tts-audiobook-tool")
    log.info(
        "[stt] starting chunked transcription: %s (model=%s, chunk=%ss, overlap=%ss)",
        path, stt_variant.id, CHUNK_DURATION, OVERLAP_DURATION,
    )
    previous_iteration_started_at: float | None = None
    previous_iteration_finished_at: float | None = None
    did_interrupt = False
    Interrupts().set("transcribing")
    try:
        # Eagerly load and warm the STT model in the worker before the chunk
        # loop starts, so the model-load time is not billed to the first
        # chunk and cannot be mistaken for a stalled first transcription.
        # The worker's TranscribeAudioCommand handler runs the eager
        # inference warm-up as part of its model load; a sliver of silence
        # keeps this a pure warm-up with no transcription work.
        silent_audio = np.zeros(1600, dtype=np.float32)
        _, warm_up_error = ModelWorker.transcribe_audio_blocking(
            prefs,
            silent_audio,
            language=language,
            stt_variant_id=stt_variant.id,
            cancel_check=lambda: Interrupts().did_interrupt,
            timeout_seconds=300.0,
        )
        if Interrupts().did_interrupt:
            did_interrupt = True
        elif warm_up_error:
            # Best effort only: the chunk loop carries its own timeout and
            # retry budgets, and a restarted cold worker re-covers the
            # model load, so a warm-up hiccup must not abort the run.
            log.warning(
                "[stt] model warm-up did not complete: %s", warm_up_error
            )

        if did_interrupt:
            return None

        for i, chunk in enumerate(
            _stream_audio_with_overlap(
                file_path=path,
                chunk_duration=CHUNK_DURATION,
                overlap_duration=OVERLAP_DURATION,
                cancel_check=lambda: Interrupts().did_interrupt,
            )
        ):
            iteration_started_at = time.monotonic()
            since_start = (
                "first chunk" if previous_iteration_started_at is None
                else f"{iteration_started_at - previous_iteration_started_at:.1f}s"
            )
            since_finish = (
                "first chunk" if previous_iteration_finished_at is None
                else f"{iteration_started_at - previous_iteration_finished_at:.1f}s"
            )
            log.info(
                "[stt] chunk %s starting at audio %.1fs; "
                "elapsed since previous start=%s, since previous finish=%s",
                i, time_offset, since_start, since_finish,
            )
            previous_iteration_started_at = iteration_started_at
            if Interrupts().did_interrupt:
                did_interrupt = True
                break

            s = f"{Ansi.LINE_HOME}{duration_string(time_offset)}"
            if duration_str:
                s += f" / {duration_str}"
            s += f"{Ansi.ERASE_REST_OF_LINE}"
            print(s, end="", flush=True)

            # A warm model normally takes ~1s per chunk. Native CUDA inference
            # can occasionally stop returning; hard-stop that worker and retry
            # only this chunk, preserving the earlier completed chunks.
            # The first chunk gets the full budget because it includes a model
            # load; a retry after a timeout restarts a cold worker, so it must
            # again cover the model load.
            chunk_timeout_seconds = 300.0 if i == 0 else 90.0
            transcription = None
            error = ""
            for attempt in range(2):
                transcription, error = ModelWorker.transcribe_audio_blocking(
                    prefs,
                    chunk,
                    language=language,
                    word_timestamps=True,
                    stt_variant_id=stt_variant.id,
                    cancel_check=lambda: Interrupts().did_interrupt,
                    timeout_seconds=(
                        300.0 if attempt > 0 else chunk_timeout_seconds
                    ),
                )
                if Interrupts().did_interrupt:
                    did_interrupt = True
                    break
                if error != STT_TRANSCRIPTION_TIMEOUT_ERROR:
                    break
                log.warning(
                    "[stt] chunk %s timed out at audio %.1fs (attempt %s/2); "
                    "worker stopped%s",
                    i, time_offset, attempt + 1,
                    "; retrying chunk" if attempt == 0 else "",
                )
                if attempt == 0:
                    print(f"\nTranscription stalled at {duration_string(time_offset)}; retrying chunk with a fresh worker...", flush=True)
            if did_interrupt:
                break
            if error or transcription is None:
                raise RuntimeError(
                    f"Transcription failed at audio {duration_string(time_offset)}: "
                    f"{error or 'Worker transcription failed'}"
                )
            transcribed_words = Transcriber.get_words_from_segments(
                transcription.segments  # type: ignore[arg-type]
            )
            updated_words = []
            for word in transcribed_words:
                updated_word = ConcreteWord(
                    start=word.start + time_offset,
                    end=word.end + time_offset,
                    word=word.word,
                    probability=word.probability
                )
                updated_words.append(updated_word)
            list_of_lists.append(updated_words)
            previous_iteration_finished_at = time.monotonic()
            log.info(
                "[stt] chunk %s finished in %.1fs (%s words)",
                i, previous_iteration_finished_at - iteration_started_at, len(updated_words),
            )

            time_offset += CHUNK_DURATION - OVERLAP_DURATION

        # The chunk stream stops on its own when a cancellation is noticed
        # while it is reading the next chunk, which ends this loop without the
        # break above ever running. A partial transcript must never be handed
        # to the alignment step, so treat that early end as an interruption.
        if Interrupts().did_interrupt:
            did_interrupt = True
    finally:
        Interrupts().clear()
        print()
        print()

    if did_interrupt:
        return None
    return list_of_lists


# ffmpeg is drained one bounded piece at a time so a cancellation is noticed
# while a chunk is still being read, and teardown is bounded so an ffmpeg that
# refuses to stop cannot hang the caller forever.
_PIPE_READ_BYTES = 64 * 1024
_FFMPEG_STOP_TIMEOUT_SECONDS = 5.0


def _stop_ffmpeg(process: Any, drained: bool) -> None:
    """Reap the ffmpeg child without deadlocking on its output pipe.

    ffmpeg's muxer thread blocks writing into the pipe the moment the parent
    stops draining it, and a blocked write never returns to run ffmpeg's
    signal handler.  Terminating first therefore leaves the child alive with
    ``wait()`` hanging on it indefinitely.  Closing our read end turns that
    blocked write into an error ffmpeg can act on, after which it exits.
    """
    try:
        process.stdout.close()
    except OSError:
        pass
    if not drained and process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=_FFMPEG_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _stream_audio_with_overlap(
    file_path: str,
    chunk_duration: int = 30,
    overlap_duration: int = 5,
    sample_rate: int = 16000,
    cancel_check: Callable[[], bool] | None = None,
) -> Generator[np.ndarray, None, None]:
    bytes_per_sample = 2
    chunk_stride = chunk_duration - overlap_duration

    bytes_per_chunk = int(sample_rate * bytes_per_sample * chunk_duration)
    bytes_per_stride = int(sample_rate * bytes_per_sample * chunk_stride)

    process = (
        ffmpeg.input(file_path)
        .output(
            "pipe:",
            loglevel="warning",
            format="s16le",
            ac=1,
            ar=sample_rate,
        )
        .run_async(pipe_stdout=True)
    )

    buffer = b""
    completed = False
    try:
        while True:
            bytes_to_read = bytes_per_chunk - len(buffer)
            raw_bytes_new = b""
            cancelled = False
            while len(raw_bytes_new) < bytes_to_read:
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    break
                piece = process.stdout.read(
                    min(bytes_to_read - len(raw_bytes_new), _PIPE_READ_BYTES)
                )
                if not piece:
                    break
                raw_bytes_new += piece

            if cancelled:
                # Leave `completed` unset so the teardown below terminates
                # ffmpeg instead of waiting for an EOF that will not come.
                break

            if not raw_bytes_new:
                completed = True
                break

            raw_bytes = buffer + raw_bytes_new

            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            yield audio

            if overlap_duration > 0:
                rewind_bytes = int(sample_rate * bytes_per_sample * overlap_duration)
                buffer = raw_bytes[-rewind_bytes:]
            else:
                buffer = b""
    finally:
        _stop_ffmpeg(process, drained=completed)


def _stitch_transcripts(
    chunks: List[List[Word]],
    time_similarity_threshold: float = 0.01
    ) -> List[Word]:
    final_words: List[Word] = []

    if not chunks:
        return []

    for chunk_index, word_list in enumerate(chunks):
        if not word_list:
            continue

        if not final_words:
            final_words.extend(word_list)
            continue

        ref_word_from_final: Word
        num_words_to_pop_if_merging: int

        if len(final_words) > 1:
            ref_word_from_final = final_words[-2]
            num_words_to_pop_if_merging = 1
        else:
            ref_word_from_final = final_words[-1]
            num_words_to_pop_if_merging = 1

        ref_time: float = ref_word_from_final.start

        start_search_index_in_current_chunk: int = 0
        if len(word_list) > 1:
            start_search_index_in_current_chunk = 1

        found_merge_point: bool = False
        actual_start_index_for_append_in_currently: int = -1

        for i in range(start_search_index_in_current_chunk, len(word_list)):
            candidate_word = word_list[i]
            if candidate_word.start > ref_time + time_similarity_threshold:
                actual_start_index_for_append_in_currently = i
                found_merge_point = True
                break

        if found_merge_point:
            for _ in range(num_words_to_pop_if_merging):
                if final_words:
                    final_words.pop()

            lst = word_list[actual_start_index_for_append_in_currently:]
            final_words.extend(lst)

    return final_words


class MatchInfo(NamedTuple):
    trans_index_start: int
    trans_index_end: int
    trans_text: str
    score: float


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def are_same_first_last_words(text1: str, text2: str) -> bool:
    a = text1.split(" ")
    b = text2.split(" ")
    if len(a) == 0 and len(b) == 0:
        return True
    if len(a) == 0 or len(b) == 0:
        return False
    first_a = a[0].strip()
    first_b = b[0].strip()
    last_a = a[-1].strip()
    last_b = b[-1].strip()
    return (first_a == first_b) and (last_a == last_b)


def word_list_to_string(lst: list[Word]) -> str:
    l = [item.word.strip() for item in lst]
    return " ".join(l)


def make_words_string(transcribed_words: list[Word], index: int, length: int=8) -> str:
    end_index = index + length
    end_index = min(end_index, len(transcribed_words))
    s = ""
    for i in range(index, end_index):
        s += transcribed_words[i].word + " "
    s = normalize_text(s)
    return s


MIN_MATCH_RATIO = 0.7
MAX_SKIP_WORDS_BASE = 45
MAX_SKIP_WORDS_LIMIT = 250

DEBUG = DEV and True

# Feature flag: verbose per-line alignment output (source/transcribed/score)
PRINT_VERBOSE = False
