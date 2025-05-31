from __future__ import annotations

from typing import Generator, List, NamedTuple
import ffmpeg
import numpy as np
import difflib
from typing import List, NamedTuple
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.util import *

class SttUtil:

    @staticmethod
    def make_timed_text_segments(
        text_segments: List[TextSegment],
        transcribed_words: List[TranscribedWord],
        print_info: bool=True
    ) -> List[TimedTextSegment]:
        """
        Aligns TextSegments with TranscribedWords to create TimedTextSegments.
        """
        results: List[TimedTextSegment] = []

        if not text_segments:
            return []
        if not transcribed_words: # If no transcript, all segments get 0 time
            for seg in text_segments:
                results.append(TimedTextSegment(
                    seg.text, seg.index_start, seg.index_end, 0.0, 0.0
                ))
            return results


        # Minimum SequenceMatcher ratio to consider a match
        MIN_MATCH_RATIO = 0.6

        # Max words from transcript to check for one segment. Dynamic based on segment length.
        # Example: if segment has 10 words, look ahead up to (10*1.5 + 10) = 25 transcribed words.
        # This helps balance accuracy with performance.
        MAX_LOOKAHEAD_FACTOR = 1.5

        # Increased for potentially more ASR noise
        MAX_LOOKAHEAD_CONSTANT = 20

        # Ensure we at least look at a few words for very short segments
        MIN_LOOKAHEAD = 5

        # Whisper can emit pure hallucinations for a given segment of audio,
        # so must have a value which accounts for the amount of text that can
        # hallucinated in whisper transcription chunk
        # We're currently using 30-second chunks

        # Max number of transcribed words to "skip" at the current position to find a match for the current text_segment.
        MAX_SKIPPABLE_TRANSCRIPT_WORDS_PER_SEGMENT = 75 # 30


        # Pointer for transcribed_words
        current_trans_idx = 0

        for segment_idx, segment in enumerate(text_segments):

            if print_info and results:
                printt(results[-1].pretty_string(segment_idx-1))

            norm_segment_text = normalize_text(segment.text)

            if not norm_segment_text: # Empty segment text after normalization
                results.append(TimedTextSegment(
                    segment.text, segment.index_start, segment.index_end, 0.0, 0.0
                ))
                continue

            segment_words_count = len(norm_segment_text.split())

            # Stores (start_trans_idx, end_trans_idx_inclusive, ratio) for the best match found for this segment
            # across all attempted start offsets in transcribed_words.
            overall_best_match_for_segment = None

            # Determine how far we can try to shift the start of the search in transcribed_words.
            # We search from current_trans_idx up to current_trans_idx + MAX_SKIPPABLE_TRANSCRIPT_WORDS_PER_SEGMENT.
            # The loop iterates over the offset from current_trans_idx.
            max_offset = MAX_SKIPPABLE_TRANSCRIPT_WORDS_PER_SEGMENT

            # Ensure we don't try to start searches beyond the end of transcribed_words
            # The number of possible start points to check.
            num_potential_starts = min(
                max_offset + 1, # +1 because we check offset 0 up to max_offset
                len(transcribed_words) - current_trans_idx # Don't go past available words
            )
            if num_potential_starts < 0: # current_trans_idx might be >= len(transcribed_words)
                num_potential_starts = 0

            for start_offset in range(num_potential_starts):
                actual_search_start_trans_idx = current_trans_idx + start_offset

                if actual_search_start_trans_idx >= len(transcribed_words): # Should be caught by num_potential_starts logic, but defensive
                    break

                # --- Inner logic to find best match for this segment starting at actual_search_start_trans_idx ---
                best_ratio_for_this_start_point = 0.0
                match_info_for_this_start_point = None # (start_idx, end_idx_inclusive, ratio)

                dynamic_lookahead_limit = int(segment_words_count * MAX_LOOKAHEAD_FACTOR) + MAX_LOOKAHEAD_CONSTANT
                if dynamic_lookahead_limit < MIN_LOOKAHEAD:
                    dynamic_lookahead_limit = MIN_LOOKAHEAD

                for k in range(1, dynamic_lookahead_limit + 1):
                    potential_trans_end_exclusive = actual_search_start_trans_idx + k
                    if potential_trans_end_exclusive > len(transcribed_words):
                        break

                    current_trans_word_objects = transcribed_words[actual_search_start_trans_idx : potential_trans_end_exclusive]
                    if not current_trans_word_objects: continue

                    norm_current_trans_text = " ".join(
                        normalize_text(tw.word) for tw in current_trans_word_objects
                    )
                    if not norm_current_trans_text: continue

                    matcher = difflib.SequenceMatcher(None, norm_segment_text, norm_current_trans_text)
                    ratio = matcher.ratio()

                    if ratio > best_ratio_for_this_start_point:
                        best_ratio_for_this_start_point = ratio
                        match_info_for_this_start_point = (
                            actual_search_start_trans_idx,
                            potential_trans_end_exclusive - 1,
                            ratio
                        )
                # --- End of inner logic ---

                if match_info_for_this_start_point:
                    current_ratio = match_info_for_this_start_point[2]
                    if overall_best_match_for_segment is None or \
                       current_ratio > overall_best_match_for_segment[2]:
                        overall_best_match_for_segment = match_info_for_this_start_point
                    # If ratios are identical, the loop order (smaller start_offset first) implicitly prefers earlier matches.

                # Optimization: If a very good match is found, no need to slide further for this segment.
                if overall_best_match_for_segment and overall_best_match_for_segment[2] > 0.95:
                    break # Break from the start_offset loop

            # After checking all potential start_offsets for the current segment:
            if overall_best_match_for_segment and overall_best_match_for_segment[2] >= MIN_MATCH_RATIO:
                match_start_idx, match_end_idx_inclusive, _ = overall_best_match_for_segment
                start_time = transcribed_words[match_start_idx].start
                end_time = transcribed_words[match_end_idx_inclusive].end

                # Sanity check for timestamps (should hold if transcribed_words are valid)
                if end_time < start_time:
                    # This can happen if a single transcribed word has end < start (bad data)
                    # or if the match is very short and ASR timestamps are noisy.
                    # Defaulting to 0,0 if this occurs, or could use start_time for both.
                    # For now, let's trust the input timestamps unless this is a problem.
                    # If it's a single word match, end_time might just be slightly off.
                    # A robust way: if end_time < start_time, use start_time for end_time for single word.
                    # If multiple words, this is a bigger issue.
                    # Given monotonic assumption, this should mostly occur for single word segments where tw.end < tw.start
                    if match_start_idx == match_end_idx_inclusive and transcribed_words[match_start_idx].end < transcribed_words[match_start_idx].start:
                        end_time = start_time # Make it a zero-duration event at start
                    # else if multiple words, this is a more serious data issue. For now, proceed.
                    # Or default to 0,0. Let's be slightly more conservative:
                    # If end_time is genuinely less than start_time, indicates a problem.
                    # We'll set to 0,0 as per the "no good match" fallback.
                    # However, the problem implies timestamps are monotonically increasing for *successive items*.
                    # And for a single item, end >= start.
                    # Let's assume good ASR data, so end_time >= start_time.

                results.append(TimedTextSegment(
                    segment.text, segment.index_start, segment.index_end,
                    start_time, end_time # Assuming end_time >= start_time after potential correction below
                ))
                # Advance current_trans_idx to after the consumed words for the next segment
                current_trans_idx = match_end_idx_inclusive + 1
            else:
                # No good match found for this segment, even after trying to skip some transcript words
                results.append(TimedTextSegment(
                    segment.text, segment.index_start, segment.index_end,
                    0.0, 0.0
                ))
                # IMPORTANT: current_trans_idx is NOT advanced.
                # The next text_segment will try to match from the same current_trans_idx.
                # This allows skipping over text_segments that don't align well with the current
                # part of the transcript, or if the transcript has gaps.

            # Ensure we don't get stuck if current_trans_idx reaches the end
            if current_trans_idx >= len(transcribed_words):
                # Fill remaining text_segments with 0.0 times
                for remaining_segment_idx in range(segment_idx + 1, len(text_segments)):
                    remaining_seg = text_segments[remaining_segment_idx]
                    results.append(TimedTextSegment(
                        remaining_seg.text, remaining_seg.index_start, remaining_seg.index_end, 0.0, 0.0
                    ))
                break # All transcribed words have been processed (or attempted)

        return results


    @staticmethod
    def make_timed_text_segments_2(chunks: list[str], min_shared_words: int = 5, debug: bool = False) -> str:
        """
        Intelligently stitches together a list of overlapping text chunks.

        Args:
            chunks: A list of strings, where each string is a text chunk.
            min_shared_words: The minimum number of words an overlap must have to be considered for stitching.
            debug: If True, prints detailed step-by-step information.

        Returns:
            A single string with the chunks stitched together.
        """
        if not chunks:
            return ""

        # Normalize chunks: split into words, then filter out empty word lists from empty/whitespace chunks
        processed_chunks_words = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text or not chunk_text.strip():
                if debug: print(f"Chunk {i} is empty or whitespace-only, skipping.")
                continue
            processed_chunks_words.append(chunk_text.split())

        if not processed_chunks_words:
            return ""
        if len(processed_chunks_words) == 1:
            return " ".join(processed_chunks_words[0])

        stitched_words = list(processed_chunks_words[0]) # Start with the first valid chunk's words

        for i in range(1, len(processed_chunks_words)):
            current_chunk_words = processed_chunks_words[i]

            if debug:
                print(f"\n>>> Processing chunk {i+1}/{len(processed_chunks_words)} (original index may vary due to empty chunk filtering) >>>")
                # print(f"  Stitched so far (ends): '...' + '{' '.join(stitched_words[-20:])}'")
                # print(f"  Current chunk (starts): '{' '.join(current_chunk_words[:20])}' + '...'")

            stitch_info = SttUtil._find_best_stitch_indices(stitched_words, current_chunk_words, debug=debug)

            if stitch_info['size'] >= min_shared_words:
                # Perform the stitch: words1[:a] + words2[b:]
                # This means we take words from 'stitched_words' up to index 'a' (exclusive),
                # and then append words from 'current_chunk_words' starting from index 'b' (inclusive).
                # The common part words1[a:a+size] is effectively replaced by words2[b:b+size].

                prefix_from_stitched = stitched_words[:stitch_info['a']]
                suffix_from_current = current_chunk_words[stitch_info['b']:]

                stitched_words = prefix_from_stitched + suffix_from_current
                if debug:
                    print(f"  Stitched: Kept {stitch_info['a']} words from previous, took from word {stitch_info['b']} of current.")
                    # print(f"    New stitched end: '...' + '{' '.join(stitched_words[-20:])}'")
            else:
                if debug:
                    print(f"  Overlap too short (size {stitch_info['size']} < {min_shared_words}) or no good match. Appending full chunk.")
                # Fallback: If no good overlap, just append the whole current chunk.
                # This might result in duplicated text if min_shared_words is too high
                # or if overlap is very messy.
                stitched_words.extend(current_chunk_words)

        return " ".join(stitched_words)

    @staticmethod
    def _find_best_stitch_indices(words1: list[str], words2: list[str], debug: bool = False):
        """
        Finds the best way to stitch words2 after words1 by identifying a common sequence.

        The method identifies a common block (words1[a:a+size] == words2[b:b+size])
        and suggests stitching as words1[:a] + words2[b:].

        Returns a dictionary: {'a': index in words1 where common block starts,
                            'b': index in words2 where common block starts,
                            'size': length of the common block,
                            'score': quality score of this stitch}
        """
        if not words1 or not words2: # Should not happen if called correctly
            return {'a': len(words1), 'b': 0, 'size': 0, 'score': -float('inf')}

        matcher = difflib.SequenceMatcher(None, words1, words2, autojunk=False)

        # Initialize with a default that implies no stitching (append full words2)
        # and a very low score.
        best_indices_and_score = {
            'a': len(words1),
            'b': 0,
            'size': 0,
            'score': -float('inf')
        }

        if debug:
            print(f"\n--- Finding stitch for w1 (ends '...' + '{' '.join(words1[-15:])}') and w2 (starts '{' '.join(words2[:15])}' + '...') ---")

        # get_matching_blocks() returns tuples (i, j, n)
        # where words1[i:i+n] == words2[j:j+n]
        # The last block is a dummy (len(words1), len(words2), 0).
        for block in matcher.get_matching_blocks():
            a, b, size = block.a, block.b, block.size

            if size == 0: # No match or end of useful blocks
                continue

            # tail1_len: Words in words1 *after* the common block. Should be small.
            tail1_len = len(words1) - (a + size)

            # head2_len: Words in words2 *before* the common block. Should be small.
            head2_len = b

            # Score: Higher is better.
            # Prioritize long matches (size).
            # Penalize if the match is not at the very end of words1 (tail1_len > 0).
            # Penalize if the match is not at the very start of words2 (head2_len > 0).
            # Weighting size more heavily can make it prefer substantial overlaps
            # even if they aren't perfectly flush at the ends.
            current_score = (size * 2) - tail1_len - head2_len
            # A simpler score: current_score = size - tail1_len - head2_len

            if debug:
                matched_w1_text = ' '.join(words1[a : a + size])
                print(f"  Block: a={a}, b={b}, size={size}. Match: '{matched_w1_text[:50]}...'")
                print(f"         tail1_len={tail1_len}, head2_len={head2_len}, Calculated Score={current_score}")

            # Update if this block gives a better score.
            # Tie-breaking: if scores are equal, prefer larger 'size'.
            # If still equal, prefer smaller 'b' (match starts earlier in words2).
            # If still equal, prefer larger 'a' (match uses later part of words1).
            if current_score > best_indices_and_score['score']:
                best_indices_and_score = {'a': a, 'b': b, 'size': size, 'score': current_score}
                if debug: print(f"    New best (score): {best_indices_and_score}")
            elif current_score == best_indices_and_score['score']:
                if size > best_indices_and_score['size']:
                    best_indices_and_score = {'a': a, 'b': b, 'size': size, 'score': current_score}
                    if debug: print(f"    New best (same score, better size): {best_indices_and_score}")
                elif size == best_indices_and_score['size']:
                    if b < best_indices_and_score['b']:
                        best_indices_and_score = {'a': a, 'b': b, 'size': size, 'score': current_score}
                        if debug: print(f"    New best (same score & size, better b): {best_indices_and_score}")
                    elif b == best_indices_and_score['b'] and a > best_indices_and_score['a']:
                        best_indices_and_score = {'a': a, 'b': b, 'size': size, 'score': current_score}
                        if debug: print(f"    New best (same score, size, b; better a): {best_indices_and_score}")

        if debug and best_indices_and_score['size'] > 0:
            print(f"--- Best stitch chosen: a={best_indices_and_score['a']}, b={best_indices_and_score['b']}, size={best_indices_and_score['size']}, score={best_indices_and_score['score']}")
            if best_indices_and_score['a'] < len(words1) : # to avoid index out of bounds if a = len(words1)
                print(f"    Keep from w1 (ends): '...' + '{' '.join(words1[max(0, best_indices_and_score['a']-10):best_indices_and_score['a']])}'")
            else:
                print(f"    Keep from w1: All of w1")

            print(f"    Overlap in w1: '{' '.join(words1[best_indices_and_score['a'] : best_indices_and_score['a'] + best_indices_and_score['size']])[:100]}...'")
            print(f"    Overlap in w2: '{' '.join(words2[best_indices_and_score['b'] : best_indices_and_score['b'] + best_indices_and_score['size']])[:100]}...'")
            print(f"    Append from w2 (starts): '{' '.join(words2[best_indices_and_score['b']:best_indices_and_score['b']+10])}' + '...'")

        return best_indices_and_score

    @staticmethod
    def transcribe_to_words(path: str) -> list[TranscribedWord]:
        """
        Creates a list of TranscribedWord instances by transcribing the audio at the given file path
        """
        list_of_lists = SttUtil._transcribe_stream_with_overlap(path, with_printout=True)
        words_list = SttUtil._stitch_transcripts(list_of_lists)
        return words_list

    @staticmethod
    def _transcribe_stream_with_overlap(path: str, with_printout: bool) -> list[list[TranscribedWord]]:
        """
        Transcribes audio file of any length using "stream with overlap".
        The resulting data will have overlapping data on each end,
        which will need to be reconciled/merged.
        """

        CHUNK_DURATION = 30
        OVERLAP_DURATION = 5

        whisper = Shared.get_whisper()
        list_of_lists: list[list[TranscribedWord]] = []
        time_offset = 0.0

        duration_string = ""
        if with_printout:
            value = AudioMetaUtil.get_audio_duration(path)
            if value:
                duration_string = time_string(value)
            print("Transcribing...")
            print()

        for i, chunk in enumerate(
            SttUtil._stream_audio_with_overlap(
                file_path=path,
                chunk_duration=CHUNK_DURATION,
                overlap_duration=OVERLAP_DURATION
            )
        ):
            if with_printout:
                s = f"{Ansi.LINE_HOME}{time_string(time_offset)}"
                if duration_string:
                    s += f" / {duration_string}"
                s += f"{Ansi.ERASE_REST_OF_LINE}"
                print(s, end="", flush=True)

            whisper_data = whisper.transcribe(chunk, word_timestamps=True, language=None)
            words = SttUtil.whisper_data_to_word_dicts(whisper_data, time_offset)
            list_of_lists.append(words)

            time_offset += CHUNK_DURATION - OVERLAP_DURATION

        if with_printout:
            print() # clear status printout
            print()

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
    def whisper_data_to_word_dicts(item: dict, time_offset: float) -> list[TranscribedWord]:
        """
        Assumes the existence of extra field, 'time_offset'
        """

        result = []

        for segment in item.get("segments", []):

            for whisper_word_dict in segment.get("words", []):

                text = whisper_word_dict["word"]
                start = float(whisper_word_dict["start"]) + float(time_offset)
                end = float(whisper_word_dict["end"]) + float(time_offset)
                probability = float(whisper_word_dict["probability"])

                word = TranscribedWord(text, start, end, probability)
                result.append(word)

        return result


    @staticmethod
    def _stitch_transcripts(
        chunks: List[List[TranscribedWord]],
        time_similarity_threshold: float = 0.01 # e.g., 10ms
        ) -> List[TranscribedWord]:
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
        final_words: List[TranscribedWord] = []

        if not chunks:
            return []

        for chunk_idx, word_list in enumerate(chunks):
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

            ref_word_from_final: TranscribedWord
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
            start_search_idx_in_current_chunk: int = 0
            if len(word_list) > 1:
                start_search_idx_in_current_chunk = 1 # Try to start from the second word.

            found_merge_point: bool = False
            # This will be the index in current_words_in_chunk from which to append.
            actual_start_idx_for_append_in_current: int = -1

            for i in range(start_search_idx_in_current_chunk, len(word_list)):
                candidate_word = word_list[i]
                # We are looking for the first word in the "reliable" part of current_words_in_chunk
                # (i.e., skipping its first word if possible) that starts clearly *after* the
                # "reliable" part of the previous content in final_words.
                if candidate_word.start > ref_time + time_similarity_threshold:
                    actual_start_idx_for_append_in_current = i
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
                final_words.extend(word_list[actual_start_idx_for_append_in_current:])
            else:
                # No suitable merge point found in the "reliable" part of current_words_in_chunk.
                # This implies current_words_in_chunk doesn't significantly or reliably extend
                # beyond what's already in final_words based on our rules.
                # So, we don't add anything from this chunk, preserving the existing content in final_words.
                # The previous chunk's (unreliable) last word remains.
                pass

        return final_words


class TranscribedWord(NamedTuple):
    # The transcribed word
    word: str
    # The timestamp of the start of the word in the source audio
    start: float
    # The timestamp of the end of the word in the source audio
    end: float
    # The transcription model's confidence level in the transcription, more or less
    probability: float

    def __repr__(self):
        return f"Word(word='{self.word}', start={self.start:.3f}, end={self.end:.3f} prob={self.probability:.2f})"

    @staticmethod
    def get_raw_text(words: list[TranscribedWord]) -> str:
        texts = [word.word for word in words]
        return "".join(texts)

#---

def normalize_text(text: str) -> str:
    """Converts text to lowercase, removes punctuation, and normalizes whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

