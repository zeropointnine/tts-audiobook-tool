from __future__ import annotations

from typing import Tuple
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.l import L
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.util import *

class TranscribeUtil:
    """
    Functions that compare transcription data (Whisper) to ground truth text
    """

    @staticmethod
    def is_audio_static(sound: Sound, whisper_data: dict) -> bool:
        """
        Very simple test
        We're trying to catch 'empty' sound gens that Oute in particular can produce with short prompts.
        (Oftentimes is just a pop sound)
        """
        DURATION_THRESH = 2.0
        if sound.duration > DURATION_THRESH:
            return False
        dicts = TranscribeUtil.get_flattened_whisper_word_dicts(whisper_data)
        return not dicts

    @staticmethod
    def is_word_count_fail(reference_text: str, whisper_data: dict) -> str:
        """
        Does simple word count comparison between transcribed text vs original text
        If difference is too large, returns fail reason message

        Threshold values are kept conservative here.
        """

        massaged_text = massage_for_text_comparison(reference_text)
        transcribed_text = TranscribeUtil.get_whisper_data_text(whisper_data)
        massaged_transcribed_text = massage_for_text_comparison(transcribed_text)
        words_src = len(massaged_text.split(" "))
        words_stt = len(massaged_transcribed_text.split(" "))
        words_delta = words_stt - words_src

        fail_message = ""
        if words_src <= 5:
            # Short phrase test
            abs_delta = abs(words_delta)
            if abs_delta >= 2:
                phrase = "too long" if abs_delta > 0 else "too short"
                fail_message = f"Transcription word count {phrase} (short phrase) (delta: {abs_delta})"
        else:
            # Normal test
            ratio = words_delta / words_src
            if ratio > 0.20:
                fail_message = f"Transcription word count too long (ratio: +{int(ratio*100)}%) (words: {words_src})"
            elif ratio < -0.20:
                fail_message = f"Transcription word count too short (ratio: {int(ratio*100)}%) (words: {words_src})"
        return fail_message

    @staticmethod
    def find_bad_repeats(reference_text: str, whisper_data: dict) -> set[str]:
        """
        Returns "adjacent words or phrases" found in the transcription but not the reference text
        (implying hallucinated repeated word/phrase)

        "Over-occurences" test overlaps with this, but this does pick up some occasional extra
        things and does not give false positives hardly at all so... keeping
        """

        reference_repeats = TranscribeUtil.find_repeats(reference_text)

        transcribed_text = TranscribeUtil.get_whisper_data_text(whisper_data)
        transcribed_repeats = TranscribeUtil.find_repeats(transcribed_text)

        return transcribed_repeats - reference_repeats


    @staticmethod
    def find_repeats(string: str) -> set[str]:
        """
        Finds words or phrases that repeat one or more times.
        The repeated word or phrase must be adjacent to each other.
        TODO: this returns overlapping items (but no practical harm done atm so)
        """

        string = massage_for_text_comparison(string)
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
    def num_bad_over_occurrences(reference_text: str, whisper_data: dict) -> int:
        """
        Returns a value if number of "word over-occurrences" is over a certain threshold
        """
        s = massage_for_text_comparison(reference_text)
        num_words = len(s.split(" "))
        thresh = 2 if num_words <= 10 else 3
        count = TranscribeUtil.count_word_over_occurrences(reference_text, whisper_data)
        return count if count >= thresh else 0


    @staticmethod
    def count_word_over_occurrences(reference_text: str, whisper_data: dict) -> int:
        """
        Count the number of extra occurrences of words found in transcription, basically
        This is another attempt at finding repeat phrases hallucinated by Oute in particular.
        """
        ref_counts = TranscribeUtil.count_words(reference_text)
        transcribed_text = TranscribeUtil.get_whisper_data_text(whisper_data)
        trans_counts = TranscribeUtil.count_words(transcribed_text)

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
    def count_words(string: str) -> dict[str, int]:
        string = massage_for_text_comparison(string)
        words = string.split()
        counts = {}
        for word in words:
            if not counts.get(word):
                counts[word] = 1
            else:
                counts[word] += 1
        return counts

    @staticmethod
    def get_semantic_match_end_time_trim(
        reference_text: str, whisper_data: dict, sound: Sound
    ) -> float | None:
        """
        Returns adjusted time at which sound clip should be trimmed if necessary
        """
        value = TranscribeUtil.get_semantic_match_end_time(reference_text, whisper_data)
        if not value:
            return None

        # Whisper end time is pretty unreliable,
        # which negates most of the usefulness of what this is trying to do but yea
        # TODO Consider a dsp algorithm that finds the nearest local minimum from the given timestamp

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
    def get_semantic_match_end_time(reference_text: str, whisper_data: dict, last_word_only: bool=False
    ) -> float | None:
        """
        Matches last transcription word and reference word and returns word end time
        """
        reference_words = massage_for_text_comparison(reference_text)
        reference_words = reference_words.split(" ")
        if len(reference_words) < 1:
            return None
        last_reference_word = reference_words[-1]

        # Being conservative here
        if reference_words.count(last_reference_word) > 1:
            last_word_only = True

        word_dicts = TranscribeUtil.get_flattened_whisper_word_dicts(whisper_data)
        if len( word_dicts ) < 1:
            return None

        # Iteratively snip end word until there's a match
        # Idea here is that TTS could add hallucinated words at the end
        # TODO verify this behaves correctly past first iteration using concrete bad gens
        MAX_ITERATIONS = 3 # being conservative here too
        num_iterations = 0
        last_dict = {}
        while len(word_dicts) >= 1:
            last_dict = word_dicts[-1]
            last_word = massage_for_text_comparison(last_dict["word"])
            is_match = (last_word == last_reference_word)
            if is_match:
                end_time = float( last_dict["end"] )
                return end_time
            word_dicts = word_dicts[:-1]
            num_iterations += 1
            if last_word_only or len(word_dicts) <= 1 or num_iterations >= MAX_ITERATIONS:
                return None

        return None

    @staticmethod
    def get_semantic_match_start_time_trim(
        reference_text: str, whisper_data: dict, sound: Sound
    ) -> float | None:
        """
        Returns adjusted start time at which sound clip should be trimmed if necessary
        """
        value = TranscribeUtil.get_semantic_match_start_time(reference_text, whisper_data)
        if not value:
            return None

        # Now account for potential silence after that value.
        # This could be done using "get-silences" but yea
        temp_sound = SoundUtil.trim(sound, value, None)
        offset = SilenceUtil.get_start_silence_end_time(temp_sound) or 0

        return value + offset

    @staticmethod
    def get_semantic_match_start_time(
        reference_text: str, whisper_data: dict
    ) -> float | None:
        """
        Matches first word transcribed word with first reference word etc
        Start time value from whisper is pretty reliable in that it doesn't crop word, etc

        TTS models can insert noises before first word, especially with 1-2 word prompts.
        They don't generally insert hallucinated words before first prompt word.
        """
        reference_words = massage_for_text_comparison(reference_text)
        reference_words = reference_words.split(" ")
        if len(reference_words) < 1:
            return None
        first_reference_word = reference_words[0]

        transcribed_word_dicts = TranscribeUtil.get_flattened_whisper_word_dicts(whisper_data)

        MAX_START_WORDS = 3
        for i in range(min(MAX_START_WORDS, len(transcribed_word_dicts))):
            transcribed_word_dict = transcribed_word_dicts[i]
            transcribed_word = transcribed_word_dict["word"]
            transcribed_word = massage_for_text_comparison(transcribed_word)
            is_match = (first_reference_word == transcribed_word)
            if is_match:
                value = float( transcribed_word_dict["start"] )
                return None if value == 0 else value
        return None


    @staticmethod
    def get_substring_time_range(
        reference_text: str,
        whisper_data: dict
    ) -> Tuple[float, float] | None:
        """
        Detects if "ground truth" text exists as a substring in a Whisper transcription,
        and returns its start/end timestamps.

        (Ignores case where both are equal)

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
        norm_ground_truth = massage_for_text_comparison(reference_text)
        if not norm_ground_truth:
            return None

        # Create list of dicts with "text", "start", and "end" properties
        all_whisper_words = []
        if whisper_data and 'segments' in whisper_data and isinstance(whisper_data['segments'], list):
            for segment in whisper_data['segments']:
                if segment and 'words' in segment and isinstance(segment['words'], list):
                    for word_info in segment['words']:
                        if (isinstance(word_info, dict) and
                                all(k in word_info for k in ('word', 'start', 'end')) and
                                isinstance(word_info['word'], str)):
                            try:
                                start_time = float(word_info['start'])
                                end_time = float(word_info['end'])
                                all_whisper_words.append({
                                    'text': word_info['word'],
                                    'start': start_time,
                                    'end': end_time
                                })
                            except (ValueError, TypeError) as e:
                                # Skip word if timestamps are not valid numbers
                                L.w("Bad value in whisper data, skipping: {e}")
                                continue

        if not all_whisper_words:
            return None

        num_whisper_words = len(all_whisper_words)

        for i in range(num_whisper_words):

            current_concatenated_raw_text = ""
            current_start_index = i
            current_start_time = all_whisper_words[i]['start']

            for j in range(i, num_whisper_words):
                current_concatenated_raw_text += all_whisper_words[j]['text']

                norm_segment_text = massage_for_text_comparison(current_concatenated_raw_text)

                if norm_segment_text == norm_ground_truth:

                    # Substring found

                    if current_start_index == 0 and j == num_whisper_words - 1:
                        # printt("is full match")
                        return None

                    start_time = current_start_time
                    end_time = all_whisper_words[j]['end']

                    # Adjust start and end times to help ensure we encompass the full audio clip
                    start_time += WHISPER_START_TIME_OFFSET
                    start_time = max(start_time, 0)
                    end_time += WHISPER_END_TIME_OFFSET

                    return (start_time, end_time)

                if len(norm_segment_text) > len(norm_ground_truth) and \
                not norm_segment_text.startswith(norm_ground_truth):
                    break

        return None

    @staticmethod
    def get_flattened_whisper_word_dicts(whisper_data: dict) -> list[dict]:
        word_dicts = []
        for segment in whisper_data["segments"]:
            word_dicts.extend(segment["words"])
        return word_dicts

    @staticmethod
    def get_pretty_whisper_data_string(data: dict) -> str:
        lst = TranscribeUtil.get_flattened_whisper_word_dicts(data)
        s = ""
        for i, item in enumerate(lst):
            prob = f"{ int((item["probability"] * 100)) }%"
            s += f"[{item["word"].strip()} {item["start"]}-{item["end"]} {prob}] "
        return s

    @staticmethod
    def get_whisper_data_text(whisper_data: dict) -> str:
        return whisper_data["text"].strip()



# ---

def get_whisper_data_last_end(whisper_data: dict) -> float | None:
    try:
        return float( whisper_data['segments'][-1]["end"] )
    except:
        return None

# ---

class Transcription:
    """
    TODO: wip. may not be useful.
    """
    def Transcription(self, whisper_data: dict):
        self.whisper_data = whisper_data
        self.word_dicts = TranscribeUtil.get_flattened_whisper_word_dicts(whisper_data)
