from __future__ import annotations

import math
from typing import List, Optional, Tuple
from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.dictionary_en import DictionaryEn


class TranscribeUtil:
    """
    """

    @staticmethod
    def is_word_failure(
        source: str,
        transcript: str,
        is_loose: bool, 
        language_code: str=""
    ) -> tuple[bool, int, int]:
        """
        Returns True if failed plus number of word failures
        """
        normalized_source, normalized_transcript = \
            TextNormalizer.normalize_source_and_transcript(source, transcript, language_code=language_code)
        num_words = TextUtil.get_word_count(normalized_source)
        fail_threshold = math.ceil(num_words / 10)
        if is_loose:
            fail_threshold *= 2
        num_word_fails = \
            TranscribeUtil.count_word_failures(normalized_source, normalized_transcript, language_code)
        return num_word_fails > fail_threshold, num_word_fails, fail_threshold

    @staticmethod
    def count_word_failures(
            normalized_source: str,
            normalized_transcript: str,
            language_code: str="",
            verbose=False
    ) -> int:
        """
        Compares source and transcribed text, and returns minimum number of failures as defined by:

            Match: 0
            Uncommon Word Match (1-to-1 or 1-to-2): 0 ("Free pass")
            Substitution: 1
            Deletion (Skip Source): 1
            Insertion (Skip Transcript): 1    

        Uses dynamic programming table (dp)
        """

        def p(s: str):
            if verbose:
                print(s)

        def is_match(a: str, b: str) -> bool:
            return (a == b) or sounds_the_same(a, b)

        def sounds_the_same(a: str, b: str) -> bool:
            return TextNormalizer.sounds_the_same_en(a, b) if language_code == "en" else False

        def is_uncommon_word(word: str) -> bool:
            return not DictionaryEn.has(word) if language_code == "en" else False

        p("")
        p(f"source: {normalized_source}")
        p(f"transc: {normalized_transcript}")
        p("")

        source_words = normalized_source.split()
        transcript_words = normalized_transcript.split()
        n = len(source_words)
        m = len(transcript_words)

        # dp[i][j] stores min failures to align source[:i] and transcript[:j]
        dp: List[List[float]] = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        # parent[i][j] stores (prev_i, prev_j, action_description)
        parent: List[List[Optional[Tuple[int, int, str]]]] = [[None] * (m + 1) for _ in range(n + 1)]
        
        dp[0][0] = 0

        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                
                # 1. Match or Substitution (Source matches Transcript)
                if i > 0 and j > 0:
                    s_word = source_words[i-1]
                    t_word = transcript_words[j-1]
                    
                    matches = is_match(s_word, t_word)
                    is_uncommon = is_uncommon_word(s_word)
                    
                    if matches:
                        cost = dp[i-1][j-1] + 0
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-1, "match_direct" if s_word == t_word else "match_homophone")
                    elif is_uncommon:
                        # Free pass 1-to-1
                        cost = dp[i-1][j-1] + 0
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-1, "uncommon_pass_1")
                    else:
                        # Substitution (Mismatch) -> Cost 1
                        cost = dp[i-1][j-1] + 1
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-1, "mismatch_sub")

                # 2. Uncommon Match 1-to-2
                if i > 0 and j > 1:
                    s_word = source_words[i-1]
                    if is_uncommon_word(s_word):
                        cost = dp[i-1][j-2] + 0
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-2, "uncommon_pass_2")

                # 3. Skip Source (Deletion) -> Cost 1
                if i > 0:
                    cost = dp[i-1][j] + 1
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i-1, j, "skip_source")

                # 4. Skip Transcript (Insertion) -> Cost 1
                if j > 0:
                    cost = dp[i][j-1] + 1
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i, j-1, "skip_transcript")

        # Reconstruct Path
        path = []
        curr_i, curr_j = n, m
        final_fails = int(dp[n][m])

        while curr_i > 0 or curr_j > 0:
            node = parent[curr_i][curr_j]
            if node is None:
                # Should not happen if reachable
                break
            prev_i, prev_j, action = node

            s_sub = source_words[prev_i:curr_i]
            t_sub = transcript_words[prev_j:curr_j]
            s_text = " ".join(s_sub) if s_sub else "-"
            t_text = " ".join(t_sub) if t_sub else "-"

            path.append((action, s_text, t_text))
            curr_i, curr_j = prev_i, prev_j

        path.reverse()

        current_fails = 0
        
        for action, s_text, t_text in path:
            word_info = f"'{s_text}' vs '{t_text}'"
            if action == "match_direct":
                p(f"{word_info} -> direct match")
            elif action == "match_homophone":
                p(f"{word_info} -> homophone match")
            elif action == "uncommon_pass_1":
                p(f"{word_info} -> no match but giving it a free pass because source word is uncommon")
            elif action == "uncommon_pass_2":
                p(f"{word_info} -> no match (2 words) but giving it a free pass because source word is uncommon")
            elif action == "skip_source":
                current_fails += 1
                p(f"{word_info} -> transcript out of words - fail #{current_fails}")
            elif action == "skip_transcript":
                current_fails += 1
                p(f"{word_info} -> source out of words - fail #{current_fails}")
            elif action == "mismatch_sub":
                current_fails += 1
                p(f"{word_info} -> mismatch - fail #{current_fails}")
                
        return final_fails

    # -----
    # TODO: Must revisit all below. Or drop.

    @staticmethod
    def get_semantic_match_end_time_trim(
        source: str, transcript_words: list[Word], sound: Sound, include_last_word
    ) -> float | None:
        """
        Returns adjusted time at which sound clip should be trimmed if necessary
        """
        value = TranscribeUtil.get_semantic_match_end_time(source, transcript_words, include_last_word)
        if not value:
            return None

        # Whisper end time is pretty unreliable,
        # which negates most of the usefulness of what this is trying to do but yea

        value += WHISPER_END_TIME_OFFSET
        if value >= sound.duration:
            return None

        # Account for potential silence before that value.
        # This could be done using "get-silences" but
        temp_sound = SoundUtil.trim(sound, 0, value)
        temp_sound_end = SilenceUtil.get_start_silence_end_time(temp_sound) or temp_sound.duration
        offset = (temp_sound.duration - temp_sound_end) * -1
        return value + offset

    @staticmethod
    def get_semantic_match_end_time(
        source: str, transcript: list[Word], include_last_word: bool
    ) -> float | None:
        """
        Matches last transcription word and reference word and returns word end time

        include_last_word:
            returns 'semantic end time' even if the match is the last word
            (which you might normally treat as redundant)
        """

        # TODO: Unlike with the text-only operations, does not account for discrepancies with compound words, but yea

        normalized_source = TextNormalizer.normalize_source(source)
        source_words = normalized_source.split(" ")
        if len(source_words) < 1:
            return None
        last_reference_word = source_words[-1]

        # Being conservative here
        if source_words.count(last_reference_word) > 1:
            last_word_only = True

        if len( transcript ) < 1:
            return None

        # Iteratively snip end word until there's a match
        # Idea here is that TTS could add hallucinated words at the end
        # TODO verify this behaves correctly past first iteration using concrete bad gens
        MAX_ITERATIONS = 3 # being conservative here too
        num_iterations = 0
        last_word_obj = {}
        while len(transcript) >= 1:
            last_word_obj = transcript[-1]
            last_word = TextNormalizer.normalize_common(last_word_obj.word)
            is_match = (last_word == last_reference_word)
            if is_match:
                if num_iterations == 0 and not include_last_word:
                    return None
                end_time = float(last_word_obj.end)
                return end_time
            transcript = transcript[:-1]
            num_iterations += 1
            if len(transcript) <= 1 or num_iterations >= MAX_ITERATIONS:
                return None

        return None

    @staticmethod
    def get_semantic_match_start_time_trim(
        source: str, transcript_words: list[Word], sound: Sound
    ) -> float | None:
        """
        Returns adjusted start time at which sound clip should be trimmed if necessary
        """
        value = TranscribeUtil.get_semantic_match_start_time(source, transcript_words)
        if not value:
            return None

        # Now account for potential silence after that value.
        # This could be done using "get-silences" but yea
        temp_sound = SoundUtil.trim(sound, value, None)
        offset = SilenceUtil.get_start_silence_end_time(temp_sound) or 0

        return value + offset

    @staticmethod
    def get_semantic_match_start_time(
        source: str, transcript_words: list[Word]
    ) -> float | None:
        """
        Matches first word transcribed word with first reference word etc
        Start time value from whisper is pretty reliable in that it doesn't crop word, etc

        TTS model can insert non-word noises before first word, especially with 1-2 word prompts.
        """
        normalized_source = TextNormalizer.normalize_common(source)
        normalized_source_words = normalized_source.split(" ")
        if len(normalized_source_words) < 1:
            return None
        first_reference_word = normalized_source_words[0]

        MAX_START_WORDS = 3
        for i in range(min(MAX_START_WORDS, len(transcript_words))):
            transcribed_word_obj = transcript_words[i]
            transcribed_word = transcribed_word_obj.word
            transcribed_word = TextNormalizer.normalize_common(transcribed_word)
            is_match = (first_reference_word == transcribed_word)
            if is_match:
                value = float( transcribed_word_obj.start )
                return None if value == 0 else value
        return None

    @staticmethod
    def get_substring_time_range(
        source: str,
        transcript_words: list[Word]
    ) -> tuple[float, float] | None:
        """
        Detects if reference text exists as a substring in a Whisper transcription,
        and returns its start/end timestamps (but ignores case where both are equal).

        The matching is done by normalizing both the ground truth text and segments of the
        transcribed text using the massage_for_comparison function. It looks for an exact
        match of a sequence of transcribed words to the ground truth phrase.

        Args:
            whisper_json: The JSON output from Whisper, expected to have 'segments'
                        each containing 'words' with 'word', 'start', and 'end' keys.
            ground_truth_text: The text phrase to search for.

        Returns:
            A tuple (start_timestamp, end_timestamp) if the phrase is found,
            otherwise None.
        """
        normalized_source = TextNormalizer.normalize_common(source)
        if not normalized_source:
            return None
        if not transcript_words:
            return None

        num_transcript_words = len(transcript_words)

        for i in range(num_transcript_words):

            current_concatenated_raw_text = ""
            current_start_index = i
            current_start_time = transcript_words[i].start

            for j in range(i, num_transcript_words):
                current_concatenated_raw_text += transcript_words[j].word

                norm_segment_text = TextNormalizer.normalize_common(current_concatenated_raw_text)

                if norm_segment_text == normalized_source:

                    # Substring found

                    if current_start_index == 0 and j == num_transcript_words - 1:
                        # printt("is full match")
                        return None

                    start_time = current_start_time
                    end_time = transcript_words[j].end

                    # Adjust start and end times to help ensure we encompass the full audio clip
                    start_time += WHISPER_START_TIME_OFFSET
                    start_time = max(start_time, 0)
                    end_time += WHISPER_END_TIME_OFFSET

                    return (start_time, end_time)

                if len(norm_segment_text) > len(normalized_source) and \
                not norm_segment_text.startswith(normalized_source):
                    break

        return None
