from __future__ import annotations

from enum import Enum, auto
import time
from pathlib import Path
from typing import Tuple, Optional, cast
import soundfile as sf
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.util import *

class ValidateUtil:

    @staticmethod
    def validate_items(
        items: list[ValidateItem],
    ) -> None:
        """
        For each time, tags validated items with "pass", prints error info for items that fail.
        Prints counts at end.

        Inits whisper at start, and unloads it at end.
        Returns True if aborted by control-c
        """

        start_time = time.time()
        num_analysed = 0
        num_detected = 0 # including ones that get corrected
        num_deleted = 0
        num_corrected = 0
        num_validated = 0

        Shared.mode = "validating"

        for item in items:

            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                break

            # Do test
            result, fail_message = ValidateUtil.validate_item(
                item=item,
                should_fix_or_delete=False,
                whisper_model=Shared.get_whisper()
            )
            num_analysed += 1

            item_info = f"{COL_ACCENT}[{COL_DEFAULT}{num_analysed}{COL_ACCENT}/{COL_DEFAULT}{len(items)}{COL_ACCENT}] Line index {COL_DEFAULT}{item.index}{COL_ACCENT}:\n"
            item_info += f"{COL_DIM}{item.path}\n{COL_DEFAULT}"

            if result == ValidateResult.VALIDATED_AND_TAGGED:
                num_validated += 1

            if result.is_fail_detected:
                num_detected += 1

            if result == ValidateResult.FAILED_ONLY:
                item_info += "Failed: " + fail_message + "\n"
                printt(item_info)
            if result == ValidateResult.FAILED_AND_CORRECTED:
                item_info += "Failed: " + fail_message + "\n" + "Corrected" + "\n"
                printt(item_info)
                num_corrected += 1
            if result == ValidateResult.FAILED_AND_DELETED:
                item_info += "Failed: " + fail_message + "\n"
                printt(item_info)
                num_deleted += 1

        # Done - print info
        print(f"Elapsed: {time_string(time.time() - start_time)}")
        printt()
        printt(f"{num_analysed} file/s analysed")
        printt(f"{num_validated} file/s marked as 'verified'")
        printt(f"{num_detected} error/s detected")
        printt()

        Shared.clear_whisper()

    @staticmethod
    def validate_item(
        item: ValidateItem,
        should_fix_or_delete: bool,
        whisper_model
    ) -> tuple[ValidateResult, str]:
        """
        """

        whisper_data = whisper_model.transcribe(item.path, word_timestamps=True, language=None)
        if not whisper_data or not "text" in whisper_data or not isinstance(whisper_data["text"], str):
            return ValidateResult.NOOP, "{COL_ERROR}Unexpected whisper result, skipping item"

        item.transcribed_text = whisper_data["text"].strip()

        # [1] Do "substring" test (and potentially fix)
        if should_fix_or_delete:
            substring_test_result = ValidateUtil.detect_is_substring_and_fix(item, cast(dict, whisper_data))
            if substring_test_result:
                did_save, error_message = substring_test_result
                if did_save:
                    return ValidateResult.FAILED_AND_CORRECTED, error_message
                else:
                    return ValidateResult.NOOP, f"{COL_ERROR}Couldn't save corrected FLAC file"
        else:
            timestamps = ValidateUtil.detect_is_substring(item, cast(dict, whisper_data))
            if timestamps:
                return ValidateResult.FAILED_ONLY, f"Excess audio detected, but substring exists at {timestamps[0]:.2f}-{timestamps[1]:.2f}"

        # [2] Do word count test
        fail_reason = ValidateUtil.is_word_count_fail(item)
        if fail_reason:
            if should_fix_or_delete:
                try:
                    Path(item.path).unlink()
                    return ValidateResult.FAILED_AND_DELETED, fail_reason
                except:
                    return ValidateResult.NOOP, "Detected error but couldn't delete file"
            else:
                return ValidateResult.FAILED_ONLY, fail_reason

        # [3] Static audio test
        is_static = ValidateUtil.is_audio_static(item, whisper_data)
        if is_static:
            fail_reason = "Audio is silent"
            if should_fix_or_delete:
                try:
                    Path(item.path).unlink()
                    return ValidateResult.FAILED_AND_DELETED, fail_reason
                except:
                    return ValidateResult.NOOP, "Detected error but couldn't delete file"
            else:
                return ValidateResult.FAILED_ONLY, fail_reason


        # At this point we consider the item to have "passed"
        path_str = item.path
        new_path_str = insert_bracket_tag_file_path(path_str,"pass")
        try:
            path = Path(path_str)
            new_path = Path(new_path_str)
            path.rename(new_path)
        except Exception:
            L.w(f"Couldn't rename {path_str} to {new_path_str}") # meh
        return ValidateResult.VALIDATED_AND_TAGGED, ""

    @staticmethod
    def detect_is_substring_and_fix(item: ValidateItem, whisper_data: dict) -> tuple[bool, str] | None:
        """
        Returns None if no action needed
        Returns
            [0] True if trim was successful
            [1] Messaging
        """
        timestamps = ValidateUtil.detect_is_substring(item, whisper_data)
        if not timestamps:
            return None

        start_time, end_time = timestamps

        message = f"Excess audio detected, but substring exists at {start_time:.2f}-{end_time:.2f}"

        # Make trimmed copy of audio file with updated filename and delete old version
        old_path = item.path
        new_path = insert_bracket_tag_file_path(item.path, "pass")
        trim_success = ConcatUtil.trim_flac_file(old_path, new_path, start_time, end_time)
        if not trim_success:
            return False, message

        item.path = new_path
        try:
            Path(old_path).unlink()
        except:
            L.w(f"Couldn't delete original file {old_path}")
        return True, message

    @staticmethod
    def is_word_count_fail(item: ValidateItem) -> str:
        """
        Does simple word count comparison between transcribed text vs original text
        If difference is too large, returns fail reason message
        """
        if not item.transcribed_text:
            return ""

        massaged_text = massage_for_text_comparison(item.text)
        transcribed_text = cast(str, item.transcribed_text)
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
    def detect_is_substring(
        item: ValidateItem,
        whisper_data: dict,
    ) -> Optional[Tuple[float, float]]:
        """
        Detects if "ground truth" text exists as a substring in a Whisper transcription and returns its start/end timestamps.
        Ignores case where ground truth text exactly maches transcription text.

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
        norm_ground_truth = massage_for_text_comparison(item.text)
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
                    START_OFFSET = -0.1 # For good measure
                    END_OFFSET = 0.25 # Because whisper consistently reports end timestamp too early
                    start_time += START_OFFSET
                    start_time = max(start_time, 0)
                    end_time += END_OFFSET

                    return (start_time, end_time)

                if len(norm_segment_text) > len(norm_ground_truth) and \
                not norm_segment_text.startswith(norm_ground_truth):
                    break

        return None

    @staticmethod
    def is_audio_static(validateItem: ValidateItem, whisper_data: dict) -> bool:
        # Test for Oute issue with very short prompts, static output

        DURATION_THRESH = 2.0

        end_time = ValidateUtil.get_whisper_data_last_end(whisper_data)
        if end_time is not None and end_time > DURATION_THRESH:
            return False

        audio, sr = sf.read(validateItem.path, dtype="float32")
        duration = len(audio) / sr
        if duration > DURATION_THRESH:
            return False

        is_first = True
        num_changes = 0
        last_value = 0

        for item in audio:
            item = round(item, 1)
            if is_first:
                is_first = False
                last_value = item
                continue
            if item != last_value:
                num_changes += 1
                if num_changes > 5:
                    return False
            last_value = item
        return True

    @staticmethod
    def get_whisper_data_last_end(whisper_data: dict) -> float | None:
        try:
            return float( whisper_data['segments'][-1]["end"] )
        except:
            return None

# ---

class ValidateItem:
    def __init__(self, i: int, path: str, text: str):
        self.index = i
        self.path = path
        self.text = text
        self.transcribed_text: str = ""

        duration = AudioMetaUtil.get_audio_duration(self.path)
        if not isinstance(duration, float):
            L.w(f"Couldn't get duration for {self.path}")
            duration = -1
        self.duration = duration

class ValidateResult(Enum):
    # File passed tests, and the file name tagged with "[pass]"
    VALIDATED_AND_TAGGED = auto()
    # File failed a test but no further action was taken
    FAILED_ONLY = auto()
    # File failed a test but was able to be corrected/edited (re-saved), file named tagged with "[pass]"
    FAILED_AND_CORRECTED = auto()
    # File failed a test and was deleted (with expectation that it will be regenerated)
    FAILED_AND_DELETED = auto()
    # Couldn't run test for whatever reason (eg, file error or smth)
    NOOP = auto()

    @property
    def is_fail_detected(self) -> bool:
        return self == ValidateResult.FAILED_ONLY or self == ValidateResult.FAILED_AND_CORRECTED or self == ValidateResult.FAILED_AND_DELETED
