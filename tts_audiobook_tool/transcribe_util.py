from __future__ import annotations

from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.l import L
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.words_dict import Dictionary


class TranscribeUtil:
    """
    Functions that compare transcription data (Whisper) to ground truth text
    """

    @staticmethod
    def is_audio_static(sound: Sound, transcribed_text: str) -> bool:
        """
        Very simple test
        We're trying to catch 'empty' sound gens that Oute in particular can produce with short prompts.
        (Oftentimes is just a pop sound)
        """
        DURATION_THRESH = 2.0
        if sound.duration > DURATION_THRESH:
            return False

        transcribed_text = TextUtil.massage_for_text_comparison(transcribed_text)
        return not transcribed_text

    @staticmethod
    def is_word_count_fail(reference_text: str, transcribed_text: str) -> str:
        """
        Does simple word count comparison between transcribed text vs original text
        If difference is too large, returns fail reason message

        Threshold values are kept conservative here.
        """
        reference_text = TextUtil.massage_for_text_comparison(reference_text)
        transcribed_text = TextUtil.massage_for_text_comparison(transcribed_text)
        num_ref_words = len(reference_text.split(" "))
        num_trans_words = len(transcribed_text.split(" "))
        words_delta = num_trans_words - num_ref_words

        fail_message = ""
        if num_ref_words <= 5:
            # Short phrase test
            abs_delta = abs(words_delta)
            if abs_delta >= 2:
                phrase = "too long" if abs_delta > 0 else "too short"
                fail_message = f"Transcription word count {phrase} (short phrase) (delta: {abs_delta})"
        else:
            # Normal test
            ratio = words_delta / num_ref_words
            if ratio > 0.20:
                fail_message = f"Transcription word count too long (ratio: +{int(ratio*100)}%) (words: {num_ref_words})"
            elif ratio < -0.20:
                fail_message = f"Transcription word count too short (ratio: {int(ratio*100)}%) (words: {num_ref_words})"
        return fail_message

    @staticmethod
    def find_bad_repeats(reference_text: str, transcribed_text: str) -> set[str]:
        """
        Returns "adjacent words or phrases" found in the transcription but not the reference text
        (implying hallucinated repeated word/phrase)

        "Over-occurences" test overlaps with this, but this does pick up some occasional extra
        things and does not give false positives hardly at all so... keeping
        """
        reference_repeats = TranscribeUtil.find_repeats(reference_text)
        transcribed_repeats = TranscribeUtil.find_repeats(transcribed_text)
        return transcribed_repeats - reference_repeats


    @staticmethod
    def find_repeats(string: str) -> set[str]:
        """
        Finds words or phrases that repeat one or more times.
        The repeated word or phrase must be adjacent to each other.
        TODO: this returns overlapping items (but no practical harm done atm so)
        """

        string = TextUtil.massage_for_text_comparison(string)
        words = string.split()

        # Not enough words for any possible repeat
        if len(words) < 2:
            return set()

        found_repeats = set()
        n = len(words)

        # Iterate through all possible phrase lengths
        # The longest possible repeating phrase can only be half the total length
        for phrase_len in range(1, (n // 2) + 1):

            # Slide a window across the list of words to find adjacent phrases
            for i in range(n - 2 * phrase_len + 1):
                phrase1 = words[i : i + phrase_len]
                phrase2 = words[i + phrase_len : i + 2 * phrase_len]

                # If two adjacent phrases are identical, add to our set
                if phrase1 == phrase2:
                    found_repeats.add(" ".join(phrase1))

        return found_repeats

    @staticmethod
    def num_bad_over_occurrences(reference_text: str, transcribed_text: str) -> int:
        """
        Returns a value if number of "word over-occurrences" is over a certain threshold
        """
        s = TextUtil.massage_for_text_comparison(reference_text)
        num_words = len(s.split(" "))
        thresh = 2 if num_words <= 10 else 3
        count = TranscribeUtil.count_word_over_occurrences(reference_text, transcribed_text)
        return count if count >= thresh else 0


    @staticmethod
    def count_word_over_occurrences(reference_text: str, transcribed_text: str) -> int:
        """
        Count the number of extra occurrences of words found in transcription, basically
        This is another attempt at finding repeat phrases hallucinated by Oute in particular.
        """
        reference_text = TextUtil.massage_for_text_comparison(reference_text)
        transcribed_text = TextUtil.massage_for_text_comparison(transcribed_text)

        ref_counts = TranscribeUtil.get_word_counts(reference_text)
        trans_counts = TranscribeUtil.get_word_counts(transcribed_text)

        count = 0
        for ref_word in ref_counts:
            if trans_counts.get(ref_word):
                # Both dicts have word
                ref_count = ref_counts[ref_word]
                trans_count = trans_counts[ref_word]
                delta = trans_count - ref_count
                if delta > 0:
                    count += delta
        return count
        # FYI, factoring in probability property is almost useful but overall isn't

    @staticmethod
    def get_word_counts(string: str) -> dict[str, int]:
        string = TextUtil.massage_for_text_comparison(string)
        words = string.split()
        counts = {}
        for word in words:
            if not counts.get(word):
                counts[word] = 1
            else:
                counts[word] += 1
        return counts

    @staticmethod
    def is_drop_fail_tail(source_text: str, trans_text: str):
        """
        Tries to determine if generated audio has missing phrase at the very end.
        Occurs especially with Fish.

        Because the current logic is "Dictionary" dependent, should only used when project language is en.
        """

        # To be considered a fail:
        # Source text - need to have 2 words of the last three that are dictionary words
        # Trans text  - must not have ANY of those 2 words in its last 4 words

        source_text = TextUtil.massage_for_text_comparison(source_text)
        trans_text = TextUtil.massage_for_text_comparison(trans_text)

        source_words = source_text.split(" ")
        trans_words = trans_text.split(" ")
        if len(trans_words) >= len(source_words):
            return False
        if len(source_words) < 4 or len(trans_words) < 4:
            return False # Keep things simple

        tail_source_words = source_words[-3:]
        dict_words = []
        for i in reversed(range(len(tail_source_words))):
            word = tail_source_words[i]
            if word in Dictionary.words:
                dict_words.append(word)
                if len(dict_words) >= 2:
                    break
        if len(dict_words) < 2:
            return False

        tail_trans_words = trans_words[-4:]
        for word in dict_words:
            if word in tail_trans_words:
                return False

        return True

    @staticmethod
    def is_drop_fail_head(source_text: str, trans_text: str):
        """
        Tries to determine if generated audio has missing phrase at the beginning.
        This is not common as "drop fail tail" above.
        WIP: Not currently used; unlike tail, same logic yields almost all false positives at least with __
        """
        source_text = TextUtil.massage_for_text_comparison(source_text)
        trans_text = TextUtil.massage_for_text_comparison(trans_text)

        source_words = source_text.split(" ")
        trans_words = trans_text.split(" ")
        if len(trans_words) >= len(source_words):
            return False
        if len(source_words) < 4 or len(trans_words) < 4:
            return False # Keep things simple

        head_source_words = source_words[:3]
        dict_words = []
        for i in range(len(head_source_words)):
            word = head_source_words[i]
            if word in Dictionary.words:
                dict_words.append(word)
                if len(dict_words) >= 2:
                    break
        if len(dict_words) < 2:
            return False

        head_trans_words = trans_words[:4]
        for word in dict_words:
            if word in head_trans_words:
                return False

        # print(source_text)
        # print(trans_text)
        # print(dict_words, head_trans_words)

        return True

    @staticmethod
    def get_semantic_match_end_time_trim(
        reference_text: str, transcribed_words: list[Word], sound: Sound, include_last_word
    ) -> float | None:
        """
        Returns adjusted time at which sound clip should be trimmed if necessary
        """
        value = TranscribeUtil.get_semantic_match_end_time(reference_text, transcribed_words, include_last_word)
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
        reference_text: str, transcribed_words: list[Word], include_last_word: bool
    ) -> float | None:
        """
        Matches last transcription word and reference word and returns word end time

        include_last_word:
            returns 'semantic end time' even if the match is the last word
            (which you might normally treat as redundant)
        """
        reference_words = TextUtil.massage_for_text_comparison(reference_text)
        reference_words = reference_words.split(" ")
        if len(reference_words) < 1:
            return None
        last_reference_word = reference_words[-1]

        # Being conservative here
        if reference_words.count(last_reference_word) > 1:
            last_word_only = True

        if len( transcribed_words ) < 1:
            return None

        # Iteratively snip end word until there's a match
        # Idea here is that TTS could add hallucinated words at the end
        # TODO verify this behaves correctly past first iteration using concrete bad gens
        MAX_ITERATIONS = 3 # being conservative here too
        num_iterations = 0
        last_word_obj = {}
        while len(transcribed_words) >= 1:
            last_word_obj = transcribed_words[-1]
            last_word = TextUtil.massage_for_text_comparison(last_word_obj.word)
            is_match = (last_word == last_reference_word)
            if is_match:
                if num_iterations == 0 and not include_last_word:
                    return None
                end_time = float(last_word_obj.end)
                return end_time
            transcribed_words = transcribed_words[:-1]
            num_iterations += 1
            if len(transcribed_words) <= 1 or num_iterations >= MAX_ITERATIONS:
                return None

        return None

    @staticmethod
    def get_semantic_match_start_time_trim(
        reference_text: str, transcribed_words: list[Word], sound: Sound
    ) -> float | None:
        """
        Returns adjusted start time at which sound clip should be trimmed if necessary
        """
        value = TranscribeUtil.get_semantic_match_start_time(reference_text, transcribed_words)
        if not value:
            return None

        # Now account for potential silence after that value.
        # This could be done using "get-silences" but yea
        temp_sound = SoundUtil.trim(sound, value, None)
        offset = SilenceUtil.get_start_silence_end_time(temp_sound) or 0

        return value + offset

    @staticmethod
    def get_semantic_match_start_time(
        reference_text: str, transcribed_words: list[Word]
    ) -> float | None:
        """
        Matches first word transcribed word with first reference word etc
        Start time value from whisper is pretty reliable in that it doesn't crop word, etc

        TTS model can insert non-word noises before first word, especially with 1-2 word prompts.
        """
        reference_words = TextUtil.massage_for_text_comparison(reference_text)
        reference_words = reference_words.split(" ")
        if len(reference_words) < 1:
            return None
        first_reference_word = reference_words[0]

        MAX_START_WORDS = 3
        for i in range(min(MAX_START_WORDS, len(transcribed_words))):
            transcribed_word_obj = transcribed_words[i]
            transcribed_word = transcribed_word_obj.word
            transcribed_word = TextUtil.massage_for_text_comparison(transcribed_word)
            is_match = (first_reference_word == transcribed_word)
            if is_match:
                value = float( transcribed_word_obj.start )
                return None if value == 0 else value
        return None


    @staticmethod
    def get_substring_time_range(
        reference_text: str,
        transcribed_words: list[Word]
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
        reference_text = TextUtil.massage_for_text_comparison(reference_text)
        if not reference_text:
            return None

        if not transcribed_words:
            return None

        num_whisper_words = len(transcribed_words)

        for i in range(num_whisper_words):

            current_concatenated_raw_text = ""
            current_start_index = i
            current_start_time = transcribed_words[i].start

            for j in range(i, num_whisper_words):
                current_concatenated_raw_text += transcribed_words[j].word

                norm_segment_text = TextUtil.massage_for_text_comparison(current_concatenated_raw_text)

                if norm_segment_text == reference_text:

                    # Substring found

                    if current_start_index == 0 and j == num_whisper_words - 1:
                        # printt("is full match")
                        return None

                    start_time = current_start_time
                    end_time = transcribed_words[j].end

                    # Adjust start and end times to help ensure we encompass the full audio clip
                    start_time += WHISPER_START_TIME_OFFSET
                    start_time = max(start_time, 0)
                    end_time += WHISPER_END_TIME_OFFSET

                    return (start_time, end_time)

                if len(norm_segment_text) > len(reference_text) and \
                not norm_segment_text.startswith(reference_text):
                    break

        return None
