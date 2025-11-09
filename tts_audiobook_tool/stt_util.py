from __future__ import annotations

from typing import Generator, List, NamedTuple
import ffmpeg
import numpy as np
import difflib
import math
from tts_audiobook_tool.app_types import ConcreteWord, Word
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class SttUtil:

    @staticmethod
    def make_timed_text_segments(
        text_segments: List[TextSegment],
        transcribed_words: List[Word],
        print_info: bool=True
    ) -> tuple[ List[TimedTextSegment], bool ]:
        """
        Force-alignment algorithm
        Takes in source text list of TextSegments and list of transcribed Words
        to create list of TimedTextSegments.

        Returns list and if did interrupt
        """

        result: List[TimedTextSegment] = []

        if not text_segments:
            return [], False
        if not transcribed_words:
            # If no transcript, all segments get 0 time
            for seg in text_segments:
                result.append(TimedTextSegment(
                    seg.text, seg.index_start, seg.index_end, 0.0, 0.0
                ))
            return result, False

        if DEBUG:
            # Clear screen and scrollback, print the lists
            print(Ansi.CLEAR_SCREEN_AND_SCROLLBACK)
            print("\nsource text:\n")
            for i, item in enumerate(text_segments):
                print(i, item.text.strip())
            s = ""
            for item in transcribed_words:
                s += item.word + " "
            s = normalize_text(s)
            print(f"\ntranscribed text:\n{s}\n")

        SigIntHandler().set("thinking")

        debug_start_time = time.time()

        max_skip_words = MAX_SKIP_WORDS_BASE
        current_skip_span = 0 # for debugging

        # Pointer for transcribed_words
        cursor = 0

        for segment_index, segment in enumerate(text_segments):

            if SigIntHandler().did_interrupt:
                SigIntHandler().clear()
                return [], True

            segment_text_normed = normalize_text(segment.text)
            segment_num_words = len(segment_text_normed.split())

            if not segment_text_normed: # Empty segment text after normalization
                result.append(TimedTextSegment(
                    segment.text, segment.index_start, segment.index_end, 0.0, 0.0
                ))
                continue

            best_match: MatchInfo | None = None

            # Outer loop slides the transcribed word starting index
            # from `cursor` to some number of words to the right

            trans_index_start_min = cursor
            trans_index_start_max = cursor + max_skip_words
            trans_index_start_max = min(trans_index_start_max, len(transcribed_words))

            for trans_index_start in range(trans_index_start_min, trans_index_start_max):

                phrase_length_min = int(segment_num_words * 0.75) or 1
                extra = int(segment_num_words * 0.25)
                extra = max(extra, 2)
                phrase_length_max = segment_num_words + extra

                # Inner loop slides the transcribed word ending index
                # in a range relatively close to the source text segment word length

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

                    # Prepend previous word to respective text chunk
                    # to help prevent false positives on short text segments
                    # Note, this may prevent correct "free-standing" matches from passing,
                    # but still, should do more good than harm
                    modded_segment_text_normed = segment_text_normed
                    modded_trans_text_normed = trans_text_normed

                    if cursor > 0 and trans_index_start > 0:
                        previous_segment = text_segments[segment_index - 1]
                        previous_segment_word = normalize_text(previous_segment.text).split(" ")[-1]
                        previous_segment_word = normalize_text(previous_segment_word)

                        previous_trans_word = transcribed_words[cursor - 1].word
                        previous_trans_word = normalize_text(previous_trans_word)

                        if previous_segment_word and previous_trans_word:
                            modded_segment_text_normed = previous_segment_word + " " + segment_text_normed
                            modded_trans_text_normed = previous_trans_word + " " + trans_text_normed

                    matcher = difflib.SequenceMatcher(None, modded_segment_text_normed, modded_trans_text_normed)
                    score = matcher.ratio()

                    if not best_match or score > best_match.score:

                        # A better match has been found
                        best_match = MatchInfo(
                            trans_index_start=trans_index_start,
                            trans_index_end=trans_index_end,
                            trans_text=trans_text_normed,
                            score=score
                        )
                        current_skip_span = trans_index_start - trans_index_start_min


            def print_result(is_success: bool): # yes rly
                printt(f"line {segment_index + 1}/{len(text_segments)}")
                printt(f"{COL_DIM}    source text: {pretty_truncate(segment_text_normed, 50)}")
                if best_match:
                    printt(f"{COL_DIM}    transcribed: {pretty_truncate(best_match.trans_text, 50)}")
                s = f"{COL_DIM}    score: {best_match.score if best_match else 0}"
                if is_success:
                    s += f" - {COL_OK}OK"
                else:
                    s += f" - {COL_ERROR}Failed"
                printt(s)
                printt()


            if best_match and best_match.score >= MIN_MATCH_RATIO:

                # Add successful result
                start_time = transcribed_words[best_match.trans_index_start].start
                # Setting end-time to the start of last word + 1
                # rather than the end-time of the last word.
                # This is cleaner due to the very inaccurate end-time reported by whisper but yea
                if best_match.trans_index_end < len(transcribed_words):
                    end_time = transcribed_words[best_match.trans_index_end].start
                else:
                    end_time = transcribed_words[best_match.trans_index_end - 1].end
                result.append(
                    TimedTextSegment(
                        segment.text,
                        segment.index_start,
                        segment.index_end,
                        start_time,
                        end_time
                ))

                if print_info:
                    print_result(True)

                # Advance cursor
                cursor = best_match.trans_index_end # rem, "trans_index_end" is EX-clusive

                # Reset max-skip-words
                max_skip_words = MAX_SKIP_WORDS_BASE

                if DEBUG:
                    print(f"had to skip {current_skip_span} words")
                    print(f"cursor is now: [{cursor+1}] {make_words_string(transcribed_words, cursor)}")
                    print()

            else:

                # No good match found for this segment
                # Add 'empty' result, do NOT advance cursor
                result.append(
                    TimedTextSegment(segment.text, segment.index_start, segment.index_end, 0.0, 0.0)
                )

                # Ensure the always-advancing source text segments can't "outrun" the transcribed text.
                # If transcription includes a chunk of text not found in the source text (up to a point...),
                # the cursor should eventually "catch up" after a few misses.
                was_max_skip = max_skip_words
                max_skip_words += len( segment.text.split(" ") ) * 2
                max_skip_words = min(max_skip_words, MAX_SKIP_WORDS_LIMIT) # Prevent pathologically slow inner-loop

                if print_info:
                    print_result(False)

                if DEBUG:
                    print(f"scanned transcript in this range: {make_words_string(transcribed_words, cursor, was_max_skip)}")
                    print(f"cursor stays at: [{cursor+1}] ")
                    print(f"max_skip_words has increased to: {max_skip_words}")
                    print()

            # ---

            if cursor >= len(transcribed_words) - 1:
                # Reached end of transcription
                # Add remaining text_segments with zeroed timestamps
                for remaining_segment_index in range(segment_index + 1, len(text_segments)):
                    remaining_seg = text_segments[remaining_segment_index]
                    result.append(TimedTextSegment(
                        remaining_seg.text, remaining_seg.index_start, remaining_seg.index_end, 0.0, 0.0
                    ))
                break

        if DEBUG:
            elapsed_sec = (time.time() - debug_start_time)
            printt(f"\nElapsed: {elapsed_sec}\n")

            for i, item in enumerate(result):
                printt(f"{i}  {item.pretty_string}")
            printt()

        SigIntHandler().clear()
        return result, False

    @staticmethod
    def transcribe_to_words(path: str) -> list[Word] | None:
        """
        Creates a list of Word instances by transcribing the audio at the given file path
        Returns None if interrupted
        """
        list_of_lists = SttUtil._transcribe_stream_with_overlap(path)
        if list_of_lists is None: # got interrupted
            return None
        words_list = SttUtil._stitch_transcripts(list_of_lists)
        return words_list

    @staticmethod
    def _transcribe_stream_with_overlap(path: str) -> list[list[Word]] | None:
        """
        Transcribes audio file of any length using "stream with overlap".
        The resulting data will have overlapping data on each end,
        which will need to be reconciled/merged.

        Returns None if interrupted
        """

        CHUNK_DURATION = 30
        OVERLAP_DURATION = 5

        list_of_lists: list[list[Word]] = []
        time_offset = 0.0

        duration_str = ""
        value = AudioMetaUtil.get_audio_duration(path)
        if value:
            duration_str = duration_string(value)

        did_interrupt = False
        SigIntHandler().set("transcribing")

        for i, chunk in enumerate(
            SttUtil._stream_audio_with_overlap(
                file_path=path,
                chunk_duration=CHUNK_DURATION,
                overlap_duration=OVERLAP_DURATION
            )
        ):
            if SigIntHandler().did_interrupt:
                did_interrupt = True
                break

            s = f"{Ansi.LINE_HOME}{duration_string(time_offset)}"
            if duration_str:
                s += f" / {duration_str}"
            s += f"{Ansi.ERASE_REST_OF_LINE}"
            print(s, end="", flush=True)

            segments, _ = Stt.get_whisper().transcribe(chunk, word_timestamps=True, language=None)
            transcribed_words = WhisperUtil.get_words_from_segments(segments)
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

        SigIntHandler().clear()

        print() # clear status printout
        print()

        if did_interrupt:
            return None
        else:
            return list_of_lists

    @staticmethod
    def _stream_audio_with_overlap(
        file_path: str,
        chunk_duration: int = 30,  # Main chunk duration (seconds)
        overlap_duration: int = 5,  # Overlap duration (seconds)
        sample_rate: int = 16000,   # Whisper's required sample rate
    ) -> Generator[np.ndarray, None, None]:
        """
        Streams audio in overlapping chunks using FFmpeg.
        """

        bytes_per_sample = 2  # 16-bit PCM = 2 bytes per sample
        chunk_stride = chunk_duration - overlap_duration  # Time between chunk starts

        # Calculate byte positions for each chunk
        bytes_per_chunk = int(sample_rate * bytes_per_sample * chunk_duration)
        bytes_per_stride = int(sample_rate * bytes_per_sample * chunk_stride)

        # Open FFmpeg stream (16kHz mono PCM)
        process = (
            ffmpeg.input(file_path)
            .output(
                "pipe:",
                loglevel="warning",
                format="s16le",
                ac=1,
                ar=sample_rate,
            )
            .run_async(pipe_stdout=True) # , quiet=True)
        )

        # Read and yield overlapping chunks
        buffer = b""
        while True:
            # Read enough bytes for the current chunk, considering the overlap
            bytes_to_read = bytes_per_chunk - len(buffer)
            raw_bytes_new = process.stdout.read(bytes_to_read)

            if not raw_bytes_new:
                break

            raw_bytes = buffer + raw_bytes_new

            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            yield audio

            # Prepare buffer for the next chunk (overlap)
            if overlap_duration > 0:
                rewind_bytes = int(sample_rate * bytes_per_sample * overlap_duration)
                buffer = raw_bytes[-rewind_bytes:]
            else:
                buffer = b""

        # Ensure the FFmpeg process is terminated
        process.wait()


    @staticmethod
    def _stitch_transcripts(
        chunks: List[List[Word]],
        time_similarity_threshold: float = 0.01 # e.g., 10ms
        ) -> List[Word]:
        """
        Stitches together a list of Word lists, handling overlaps between chunks.

        The function assumes a rewind of a few seconds (e.g., 5s) occurs between
        chunks, causing an overlap. It also accounts for the first and last words
        of any given chunk being potentially unreliable.

        Args:
            chunks: A list of lists, where each inner list contains Word instances
                    for an audio chunk.
            time_similarity_threshold: A small float value. A word from the current
                                    chunk is considered "new" if its start time
                                    is greater than the reference time from the
                                    previous chunk plus this threshold. This accounts
                                    for minor timing jitter and ensures strict progression.

        Returns:
            A flat list of Word instances representing the stitched transcript.
        """

        final_words: List[Word] = []

        if not chunks:
            return []

        for chunk_index, word_list in enumerate(chunks):
            if not word_list:  # Skip empty chunks
                continue

            if not final_words:  # This is the first non-empty chunk
                # Add all words from the first chunk. Its first word is kept as there's
                # no prior context, and its last word will be handled if a subsequent
                # chunk overlaps and provides a better continuation.
                final_words.extend(word_list)
                continue

            # At this point, final_words is not empty, and current_words_in_chunk is not empty.
            # We need to find a merge point, respecting that first/last words of raw chunks are unreliable.

            # Determine the reference point from final_words (derived from the previously appended chunk).
            # This reference point should ideally be from a "reliable" part of final_words.
            # - final_words[-1] is the (unreliable) end of the segment from the previous chunk.
            # - If final_words has >1 item, final_words[-2] is a more reliable reference point from past content.
            # - If final_words has only 1 item, final_words[-1] is the only reference we have.

            ref_word_from_final: Word
            num_words_to_pop_if_merging: int

            if len(final_words) > 1:
                # Use the second-to-last word as the reliable anchor from previous content.
                # Its start time marks a point we are confident about.
                ref_word_from_final = final_words[-2]
                # If we merge, we'll pop the last (unreliable) word from final_words,
                # as the new chunk will provide words for that (now later) time period.
                num_words_to_pop_if_merging = 1
            else:  # final_words has only 1 word.
                # This single word is our only anchor. It's both "first" and "last".
                ref_word_from_final = final_words[-1]
                # If we merge, we'll pop this single word, to be replaced by new chunk's content.
                num_words_to_pop_if_merging = 1

            ref_time: float = ref_word_from_final.start

            # Determine where to start searching in current_words_in_chunk.
            # Ideally, skip its first (unreliable) word to find a more reliable start.
            # If current_words_in_chunk has only 1 word, we must consider it.
            start_search_index_in_current_chunk: int = 0
            if len(word_list) > 1:
                start_search_index_in_current_chunk = 1 # Try to start from the second word.

            found_merge_point: bool = False
            # This will be the index in current_words_in_chunk from which to append.
            actual_start_index_for_append_in_currently: int = -1

            for i in range(start_search_index_in_current_chunk, len(word_list)):
                candidate_word = word_list[i]
                # We are looking for the first word in the "reliable" part of current_words_in_chunk
                # (i.e., skipping its first word if possible) that starts clearly *after* the
                # "reliable" part of the previous content in final_words.
                if candidate_word.start > ref_time + time_similarity_threshold:
                    actual_start_index_for_append_in_currently = i
                    found_merge_point = True
                    break

            if found_merge_point:
                # A suitable merge point was found.
                # Remove the end segment of the previously appended chunk from final_words.
                # This typically means removing the last (unreliable) word from the prior segment.
                for _ in range(num_words_to_pop_if_merging):
                    if final_words: # Should always be true here given the logic flow
                        final_words.pop()

                # Add the new words from current_words_in_chunk, starting from the identified merge point.
                lst = word_list[actual_start_index_for_append_in_currently:]
                final_words.extend(lst)
            else:
                # No suitable merge point found in the "reliable" part of current_words_in_chunk.
                # This implies current_words_in_chunk doesn't significantly or reliably extend
                # beyond what's already in final_words based on our rules.
                # So, we don't add anything from this chunk, preserving the existing content in final_words.
                # The previous chunk's (unreliable) last word remains.
                pass

        return final_words

# ---

class MatchInfo(NamedTuple):
    trans_index_start: int
    trans_index_end: int # rem, is exclusive
    trans_text: str # used for debugging
    score: float

def normalize_text(text: str) -> str:
    """Converts text to lowercase, removes punctuation, and normalizes whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def are_same_first_last_words(text1: str, text2: str) -> bool:
    """
    Quick 'shorthand' test for force-alignment result,,
    used to id potential issues for debugging purposes
    """
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

def pretty_truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return f"{COL_DEFAULT}{text}"
    width -= 3 # bc triple-dots
    a_len = math.ceil(width / 2)
    b_len = math.floor(width / 2)
    a = text[:a_len]
    b = text[-b_len:]
    return f"{COL_DEFAULT}{a}{COL_DIM}...{COL_DEFAULT}{b}"

def word_list_to_string(lst: list[Word]) -> str:
    l = [item.word.strip() for item in lst]
    return " ".join(l)

def make_words_string(transcribed_words: list[Word], index: int, length: int=8) -> str:
    """ used for debugging"""
    end_index = index + length
    end_index = min(end_index, len(transcribed_words))
    s = ""
    for i in range(index, end_index):
        s += transcribed_words[i].word + " "
    s = normalize_text(s)
    return s

# ---

# Minimum SequenceMatcher ratio to consider a match:
# Higher values mean less false positives on short chunks,
# especially in extraneous text at the beginning of the text file (eg publisher info),
# but will miss more real matches on short chunks.
MIN_MATCH_RATIO = 0.7

# Max number of transcribed words to "skip" at the current position
# to find a match for the current text_segment.
# Should be at least as long as source text word length per segment
MAX_SKIP_WORDS_BASE = 45
MAX_SKIP_WORDS_LIMIT = 250

DEBUG = DEV and True
