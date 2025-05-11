from __future__ import annotations

import gc
import time
import whisper
from pathlib import Path
import torch
from typing import Tuple, Optional, cast
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class VerifyUtil:

    @staticmethod
    def verify(state: State) -> None:

        printt(f"{COL_ACCENT}Verify generated audio:\n")

        all_items = VerifyUtil.make_items(state)
        if not all_items:
            ask("No generated items. Press enter: ")
            return

        avg_sec_per_char = calc_avg_sec_per_char(all_items)

        items = [item for item in all_items if not item.already_verified]
        if not items:
            ask("All generated audio files already verified. Press enter:" )
            return

        num_already_verified = len(all_items) - len(items)

        if not num_already_verified:
            printt("We will analyse the generated audio lines and make a 'best guess' as to if they have any significant errors.")
            printt("")
            printt("Lines with errors that can be resolved by simply trimming the audio will be automatically fixed.")
            printt("")
            printt(f"Other lines with suspected errors {COL_ACCENT}will be deleted{COL_DEFAULT}, and will need to be regenerated from the main menu. Regenerating these lines offers a fair possibility of resolving these errors.")
            printt("")

        printt(f"Total lines generated: {len(all_items)}")
        if num_already_verified:
            printt(f"Lines already marked as verified: {num_already_verified}")
        printt(f"Lines left to be analysed: {len(items)}")
        printt()

        num_missing = len(state.text_segments) - len(all_items)
        if num_missing > 0:
            printt(f"Note that {num_missing} lines from the project still need to be generated.")
            printt()
        printt(f"{make_hotkey_string("1")} Start (trim or delete files as needed)")
        printt(f"{make_hotkey_string("2")} Verify only (don't trim or delete)")
        printt()
        hotkey = ask_hotkey()
        if hotkey == "1":
            fix_and_delete = True
        elif hotkey == "2":
            fix_and_delete = False
        else:
            return

        VerifyUtil.do_tests(items, avg_sec_per_char, fix_and_delete=fix_and_delete)

        ask("Finished. Press enter: ")

    @staticmethod
    def do_tests(
        items: list[Item],
        avg_sec_per_char: float, # used for duration test
        fix_and_delete: bool
    ) -> None:
        """
        Uses whisper to do some kind of a text content comparisons between source text and transcribed text
        Sets "result" on items.
        When there is an opportunity to fix a detected error by simply trimming audio sample, does so.
        Renames files that pass tests with "[pass]"
        """

        start_time = time.time()

        whisper_model = None
        num_detected = 0 # including ones that get corrected
        num_deleted = 0
        num_corrected = 0
        num_verified = 0

        for item in items:

            # [0] Do transcription
            if not whisper_model:
                printt("Initializing whisper model...")
                printt()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                whisper_model = whisper.load_model("turbo", device=device)
                printt("Starting")
                printt()

            whisper_data = whisper_model.transcribe(item.path, word_timestamps=True, language=None)
            if not whisper_data or not "text" in whisper_data or not isinstance(whisper_data["text"], str):
                printt(f"{COL_ERROR}Unexpected whisper result, skipping item")
                continue
            item.transcribed_text = whisper_data["text"].strip()

            # [1] Do "substring" test
            if fix_and_delete:
                substring_test_result = VerifyUtil.detect_is_substring_and_fix(item, cast(dict, whisper_data))
                if substring_test_result:
                    num_detected += 1
                    did_save, message = substring_test_result
                    if did_save:
                        num_corrected += 1
                        action_taken = "Fixed"
                    else:
                        action_taken = f"{COL_ERROR}Couldn't save new FLAC file"
                    print_item_info(item, message, action_taken)
                    continue
            else:
                timestamps = VerifyUtil.detect_is_substring(item, cast(dict, whisper_data))
                if timestamps:
                    num_detected += 1
                    message = f"Excess audio detected, substring exists at {timestamps[0]:.2f}-{timestamps[1]:.2f}"
                    print_item_info(item, message)
                    continue

            # [2] Do word count test
            fail_reason = VerifyUtil.is_word_count_fail(item)
            if fail_reason:
                num_detected += 1
                if fix_and_delete:
                    try:
                        Path(item.path).unlink()
                        num_deleted += 1
                        action_taken = "Deleted"
                    except:
                        action_taken = f"{COL_ERROR}Couldn't delete"
                else:
                    action_taken = ""
                print_item_info(item, fail_reason, action_taken)
                continue

            # [3] Do duration test, as a bonus (may or may not fill in some cracks)
            if avg_sec_per_char > 0:
                fail_reason = VerifyUtil.is_duration_fail(item, avg_sec_per_char)
                if fail_reason:
                    num_detected += 1
                    if fix_and_delete:
                        try:
                            Path(item.path).unlink()
                            num_deleted += 1
                            action_taken = "Deleted"
                        except:
                            action_taken = f"{COL_ERROR}Couldn't delete"
                    else:
                        action_taken = ""
                    print_item_info(item, fail_reason, action_taken)
                    continue

            # At this point we consider the item to have "passed"
            path_str = item.path
            new_path_str = insert_bracket_tag_file_path(path_str,"pass")
            try:
                path = Path(path_str)
                new_path = Path(new_path_str)
                path.rename(new_path)
                num_verified += 1
            except Exception as e:
                L.w(f"Couldn't rename {path_str} to {new_path_str}")

        # Done
        print(f"Elapsed: {AppUtil.time_string(time.time() - start_time)}")
        printt()
        printt(f"{len(items)} files analysed")
        printt(f"{num_verified} files marked as 'verified'")
        printt(f"{num_detected} error/s detected")
        if fix_and_delete:
            printt(f"{num_corrected} files corrected")
            printt()
            printt(f"{num_deleted} files were deleted, and will need to be regenerated.")
        printt()

        if whisper_model:
            # Cleanup
            printt("Unloading whisper...\n")
            del whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def detect_is_substring_and_fix(item: Item, whisper_data: dict) -> tuple[bool, str] | None:
        """
        Returns None if no action needed
        Returns
            [0] True if trim was successful
            [1] Messaging
        """
        timestamps = VerifyUtil.detect_is_substring(item, whisper_data)
        if not timestamps:
            return None

        start_time, end_time = timestamps

        message = f"Excess audio detected, substring exists at {start_time:.2f}-{end_time:.2f}"

        # Make trimmed copy of audio file with updated filename and delete old version
        new_path = insert_bracket_tag_file_path(item.path, "pass")
        new_path = insert_bracket_tag_file_path(new_path, "trimmed")
        did_trim = ConcatUtil.trim_flac_file(item.path, new_path, start_time, end_time)
        if not did_trim:
            return False, message

        # Is considered success at this point
        item.path = new_path
        try:
            Path(item.path).unlink()
        except:
            L.w(f"Couldn't delete original file {item.path}")
        return True, message


    @staticmethod
    def is_word_count_fail(item: Item) -> str:
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

        fail_reason = ""
        if words_src <= 5:
            # Short phrase test
            abs_delta = abs(words_delta)
            if abs_delta >= 2:
                phrase = "too long" if abs_delta > 0 else "too short"
                fail_reason = f"Transcription word count {phrase} (short phrase) (delta: {abs_delta})"
        else:
            # Normal test
            ratio = words_delta / words_src
            if ratio > 0.20:
                fail_reason = f"Transcription word count too long (ratio: +{int(ratio*100)}%) (words: {words_src})"
            elif ratio < -0.20:
                fail_reason = f"Transcription word count too short (ratio: {int(ratio*100)}%) (words: {words_src})"
        return fail_reason

    @staticmethod
    def is_duration_fail(item: Item, avg_sec_per_char: float) -> str:
        """
        Does simple "speech rate" test between item and average.
        If difference is too large, prints info, returns fail reason message
        """
        MIN_CHARS = 30
        CHARS_A = 30
        CHARS_B = 60
        MULT_A = 2.5
        MULT_B = 1.75

        num_chars = len(item.text)
        if num_chars < MIN_CHARS:
            return ""
        sec_per_char = item.duration / num_chars
        multiplier = sec_per_char / avg_sec_per_char
        mult_thresh_max = lerp_clamped(num_chars, CHARS_A, CHARS_B, MULT_A, MULT_B) # As num chars increases, multiplier threshold decreases
        MULT_THRESH_MIN = 0.5

        fail_reason = ""
        if multiplier >= mult_thresh_max:
            fail_reason = f"Duration too long (multiplier: {multiplier:.2f} vs {mult_thresh_max:.2f})"
        elif multiplier <= MULT_THRESH_MIN:
            fail_reason = f"Duration too short (multiplier: {multiplier:.2f} vs {MULT_THRESH_MIN:.2f})"
        return fail_reason

    @staticmethod
    def make_items(state: State) -> list[Item]:
        """
        Creates items, including their audio file duration values.
        """
        items = []
        index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        for index, file_path in index_to_path.items():
            text_segment = state.text_segments[index]
            seconds = AppUtil.get_flac_file_duration(file_path)
            if not isinstance(seconds, float):
                L.w(f"Ignoring - couldn't get duration for {file_path}")
                continue
            item = Item(index, file_path, text_segment, seconds)
            item.already_verified = "[pass]" in file_path
            items.append(item)
        return items

    @staticmethod
    def detect_is_substring(
        item: Item,
        whisper_data: dict,
    ) -> Optional[Tuple[float, float]]:
        """
        Finds if a ground truth text phrase exists in a Whisper transcription and returns its start/end timestamps.

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
                            except (ValueError, TypeError):
                                # Skip word if timestamps are not valid numbers
                                continue

        if not all_whisper_words:
            return None

        n_whisper_words = len(all_whisper_words)
        for i in range(n_whisper_words):
            current_concatenated_raw_text = ""
            current_start_time = all_whisper_words[i]['start']

            for j in range(i, n_whisper_words):
                current_concatenated_raw_text += all_whisper_words[j]['text']

                norm_segment_text = massage_for_text_comparison(current_concatenated_raw_text)

                if norm_segment_text == norm_ground_truth:

                    start_time = current_start_time
                    end_time = all_whisper_words[j]['end']

                    # Adjust start and end times
                    START_OFFSET = -0.1 # For good measure
                    END_OFFSET = 0.25 # Whisper consistently reports end timestamp too early
                    start_time += START_OFFSET
                    start_time = max(start_time, 0)
                    end_time += END_OFFSET
                    if abs(item.duration - (end_time - start_time)) < 1.0:
                        # TODO: This logic should not need duration, shd be based just on text content
                        return None

                    return (start_time, end_time)

                if len(norm_segment_text) > len(norm_ground_truth) and \
                not norm_segment_text.startswith(norm_ground_truth):
                    break

        return None

# ---

def calc_avg_sec_per_char(all_items: list[Item]) -> float:
    MIN_ITEMS = 50
    if len(all_items) < MIN_ITEMS:
        return 0.0

    total_chars = 0
    total_duration = 0.0
    for item in all_items:
        total_chars += len(item.text)
        total_duration += item.duration
    return total_duration / total_chars

def print_item_info(item: Item, message: str, action_taken: str="") -> None:
    printt(f"{make_hotkey_string(str(item.index))} {item.path}")
    printt(f"{" " * 8}{message}")
    if action_taken:
        printt(f"{" " * 8}{action_taken}")
    print()

# ---

class Item:
    def __init__(self, i: int, path: str, text: str, duration: float):
        self.index = i
        self.path = path
        self.text = text
        self.duration = duration
        self.transcribed_text: str = ""
        self.already_verified = False

# with open("temp_items.pickle", 'wb') as f:
#     pickle.dump(my_object, f)

# with open("temp_items.pickle", 'rb') as f:
#     items: list[Item] = pickle.load(f)
