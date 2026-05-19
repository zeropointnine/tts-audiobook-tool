from __future__ import annotations

"""
Enhance-flow alignment helpers.

This module supports the "enhance existing audiobook" flow by:
- transcribing long audio into word timestamps using overlapping chunks,
- stitching chunk transcripts into a single word stream,
- aligning source phrases to transcript words to build TimedPhrases.
"""

from typing import Generator, List, NamedTuple
import ffmpeg
import numpy as np
import difflib
from tts_audiobook_tool.app_types import ConcreteWord, Word
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.app_types.phrase import Phrase
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import *
from tts_audiobook_tool.transcriber import Transcriber


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

    result: List[TimedPhrase] = []

    if not phrases:
        return [], False
    if not transcribed_words:
        for phrase in phrases:
            result.append(TimedPhrase(phrase.text, 0.0, 0.0))
        return result, False

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

    max_skip_words = MAX_SKIP_WORDS_BASE
    current_skip_span = 0

    cursor = 0

    for segment_index, segment in enumerate(phrases):

        if Interrupts().did_interrupt:
            Interrupts().clear()
            return [], True

        text_normed = normalize_text(segment.text)
        num_words = len(text_normed.split())

        if not text_normed:
            result.append(TimedPhrase(segment.text, 0.0, 0.0))
            continue

        best_match: MatchInfo | None = None

        trans_index_start_min = cursor
        trans_index_start_max = cursor + max_skip_words
        trans_index_start_max = min(trans_index_start_max, len(transcribed_words))

        for trans_index_start in range(trans_index_start_min, trans_index_start_max):

            phrase_length_min = int(num_words * 0.75) or 1
            extra = int(num_words * 0.25)
            extra = max(extra, 2)
            phrase_length_max = num_words + extra

            trans_index_end_min = trans_index_start + phrase_length_min
            trans_index_end_max = trans_index_start + phrase_length_max
            trans_index_end_max = min(trans_index_end_max, len(transcribed_words))

            for trans_index_end in range(trans_index_end_min, trans_index_end_max):

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

                if cursor > 0 and trans_index_start > 0:
                    previous_segment = phrases[segment_index - 1]
                    previous_segment_word = normalize_text(previous_segment.text).split(" ")[-1]
                    previous_segment_word = normalize_text(previous_segment_word)

                    previous_trans_word = transcribed_words[cursor - 1].word
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
            printt(f"line {segment_index + 1}/{len(phrases)}")
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
            result.append(
                TimedPhrase(
                    segment.text,
                    start_time,
                    end_time
            ))

            if print_info:
                print_result(True)

            cursor = best_match.trans_index_end
            max_skip_words = MAX_SKIP_WORDS_BASE

            if DEBUG:
                print(f"had to skip {current_skip_span} words")
                print(f"cursor is now: [{cursor+1}] {make_words_string(transcribed_words, cursor)}")
                print()

        else:
            result.append(
                TimedPhrase(segment.text, 0.0, 0.0)
            )

            was_max_skip = max_skip_words
            max_skip_words += len( segment.text.split(" ") ) * 2
            max_skip_words = min(max_skip_words, MAX_SKIP_WORDS_LIMIT)

            if print_info:
                print_result(False)

            if DEBUG:
                print(f"scanned transcript in this range: {make_words_string(transcribed_words, cursor, was_max_skip)}")
                print(f"cursor stays at: [{cursor+1}] ")
                print(f"max_skip_words has increased to: {max_skip_words}")
                print()

        if cursor >= len(transcribed_words) - 1:
            for remaining_segment_index in range(segment_index + 1, len(phrases)):
                remaining_seg = phrases[remaining_segment_index]
                result.append(TimedPhrase(
                    remaining_seg.text, 0.0, 0.0
                ))
            break

    if DEBUG:
        elapsed_sec = (time.time() - debug_start_time)
        printt(f"\nElapsed: {elapsed_sec}\n")

        for i, item in enumerate(result):
            printt(f"{i}  {item}")
        printt()

    Interrupts().clear()
    return result, False


def transcribe_to_words(path: str) -> list[Word] | None:
    """
    Creates a list of Word instances by transcribing the audio at the given file path
    Returns None if interrupted
    """
    list_of_lists = _transcribe_stream_with_overlap(path)
    if list_of_lists is None:
        return None
    words_list = _stitch_transcripts(list_of_lists)
    return words_list


def _transcribe_stream_with_overlap(path: str) -> list[list[Word]] | None:
    CHUNK_DURATION = 30
    OVERLAP_DURATION = 5

    list_of_lists: list[list[Word]] = []
    time_offset = 0.0

    duration_str = ""
    value = AudioMetaUtil.get_audio_duration(path)
    if value:
        duration_str = duration_string(value)

    did_interrupt = False
    Interrupts().set("transcribing")

    for i, chunk in enumerate(
        _stream_audio_with_overlap(
            file_path=path,
            chunk_duration=CHUNK_DURATION,
            overlap_duration=OVERLAP_DURATION
        )
    ):
        if Interrupts().did_interrupt:
            did_interrupt = True
            break

        s = f"{Ansi.LINE_HOME}{duration_string(time_offset)}"
        if duration_str:
            s += f" / {duration_str}"
        s += f"{Ansi.ERASE_REST_OF_LINE}"
        print(s, end="", flush=True)

        segments, _ = Stt.get_whisper().transcribe(chunk, word_timestamps=True, language=None)
        transcribed_words = Transcriber.get_words_from_segments(segments)
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

        time_offset += CHUNK_DURATION - OVERLAP_DURATION

    Interrupts().clear()

    print()
    print()

    if did_interrupt:
        return None
    else:
        return list_of_lists


def _stream_audio_with_overlap(
    file_path: str,
    chunk_duration: int = 30,
    overlap_duration: int = 5,
    sample_rate: int = 16000,
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
    while True:
        bytes_to_read = bytes_per_chunk - len(buffer)
        raw_bytes_new = process.stdout.read(bytes_to_read)

        if not raw_bytes_new:
            break

        raw_bytes = buffer + raw_bytes_new

        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        yield audio

        if overlap_duration > 0:
            rewind_bytes = int(sample_rate * bytes_per_sample * overlap_duration)
            buffer = raw_bytes[-rewind_bytes:]
        else:
            buffer = b""

    process.wait()


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
