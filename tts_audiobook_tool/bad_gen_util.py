from __future__ import annotations

import gc
import os
import pickle
import whisper
import re
import random
from pathlib import Path
import torch
from typing import Tuple, Optional, cast
from rapidfuzz import fuzz
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from .util import *

class BadGenUtil:

    @staticmethod
    def ask_detect(state: State) -> bool:
        """
        Returns True if user requests for deleted files to be regenerated
        """
        items = BadGenUtil.make_items(state)

        # with open("temp_items.pickle", 'rb') as f:
        #     items: list[Item] = pickle.load(f)

        ask("Will do quick duration test first. Press enter: ")
        BadGenUtil.do_duration_test(items)

        ask("Will do speech-to-text test next. This may take some time. Press enter: ")
        BadGenUtil.do_stt_test(items)

        bad_items = []
        for item in items:
            if isinstance(item.result, FailResult):
                bad_items.append(item)

        if len(bad_items) == 0:
            ask("No bad gens found. Press enter: ")
            return False

        b = ask_confirm(f"Press {make_hotkey_string("Y")} to delete {len(bad_items)} suspected bad generations (They will need to be regenerated): ")
        if not b:
            return False

        for item in bad_items:
            try:
                Path(item.path).unlink()
            except:
                L.w(f"Couldn't delete {item.path}")

        # with open("temp_items.pickle", 'wb') as f:
        #     pickle.dump(items, f)

        b = ask_confirm(f"Enter {make_hotkey_string("Y")} to regenerate the deleted files now (as well as any remaining text lines): ")
        return b

    @staticmethod
    def can_do_duration_test(items: list[Item]) -> bool:
        return len(items) >= 50

    @staticmethod
    def do_duration_test(items: list[Item]) -> None:
        """
        Sets items as failed that have suspiciously long or short duration relative to character count.
        Returns False if skipped because too few items
        """
        if not BadGenUtil.can_do_duration_test(items):
            return

        # Calculate average seconds per character
        total_chars = 0
        total_duration = 0.0
        for item in items:
            total_chars += len(item.text)
            total_duration += item.duration
        avg_sec_per_char = total_duration / total_chars

        # As num chars increases, multiplier threshold decreases
        MIN_CHARS = 30
        CHARS_A = 30
        CHARS_B = 60
        MULT_A = 2.5
        MULT_B = 1.75

        for item in items:

            if item.result is not None:
                continue

            num_chars = len(item.text)
            if num_chars < MIN_CHARS:
                continue

            sec_per_char = item.duration / num_chars
            multiplier = sec_per_char / avg_sec_per_char
            mult_thresh_max = lerp_clamped(num_chars, CHARS_A, CHARS_B, MULT_A, MULT_B)
            MULT_THRESH_MIN = 0.5

            if multiplier >= mult_thresh_max:
                item.result = FailResult(f"Duration too long (multiplier: {multiplier:.2f} vs {mult_thresh_max:.2f})")
            elif multiplier <= MULT_THRESH_MIN:
                item.result = FailResult(f"Duration too short (multiplier: {multiplier:.2f} vs {MULT_THRESH_MIN:.2f})")
            if isinstance(item.result, FailResult):
                BadGenUtil.print_item_fail(item)

    @staticmethod
    def do_stt_test(all_items: list[Item]) -> None:

        whisper_model = None

        # Include only items do not yet have a result
        items = all_items
        items = [item for item in items if item.result is None]
        # Include items whose filenames are not tagged with "[pass]"
        items = [item for item in items if not "[pass]" in item.path]

        for item in items:

            # Get transcription if necessary
            if isinstance(item.transcribed_text, Untranscribed):

                if not whisper_model:
                    printt("Initializing whisper model...")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    whisper_model = whisper.load_model("turbo", device=device)
                    printt()

                BadGenUtil.populate_transcribed_text(item, whisper_model)

            if isinstance(item.transcribed_text, TranscribeFail):
                continue

            massaged_text = massage_for_comparison(item.text)
            transcribed_text = cast(str, item.transcribed_text)
            massaged_transcribed_text = massage_for_comparison(transcribed_text)
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
                    fail_reason = f"Transcription too long (ratio: +{ratio:.2f}) (num words: len {words_src})"
                elif ratio < -0.20:
                    fail_reason = f"Transcription too short (ratio: {ratio:.2f}) (num words: len {words_src})"

            if fail_reason:
                # Whisper can hallucinate extra words at the end and create false positives
                # ("condition_on_previous_text=False" did not change this one way or another, even though it's supposed to.)
                # But outte has a tendency to hallucinate too, where it will repeat phrases or sentences at the end as well.
                # So these two issues overlap.
                # However the outte hallucinations repeat previous phrases but whisper ones append short unrelated ones, so.
                # It's much more important to under-detect false positives rather than the opposite, since being forced to regenerate
                # false positives is relatively harmess, versus letting a bad generation go through, which is much worse.
                if False:
                    # Work in progress
                    tup = BadGenUtil.detect_whisper_hallucination(massaged_text, massaged_transcribed_text)
                    if tup[0] == True:
                        fail_reason = ""
                        print("fail reason rescinded:", tup)

            if fail_reason:
                item.result = FailResult(fail_reason)
                BadGenUtil.print_item_fail(item)
            else:
                item.result = PassResult()
                rename_as_passed(item.path)

        if whisper_model:
            # Cleanup
            del whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        print("\a", end="")

    @staticmethod
    def detect_whisper_hallucination(
        ground_truth: str,
        whisper_text: str,
        accuracy_threshold: int = 90,
        min_hallucination_len: int = 5,
        base_match_leniency_percent: float = 0.20,
        hallucination_to_base_max_similarity_threshold: int = 40
    ) -> Tuple[bool, str, str, Optional[int]]:
        """
        Fuzzily detects if whisper_text contains ground_truth as a base and then appends
        random, dissimilar phrases (hallucinations). The dissimilarity check is performed
        after identifying the best potential base and suffix.

        Gemini 2.5 Pro, untouched

        Args:
            ground_truth (str): The correct reference text.
            whisper_text (str): The text output by Whisper.
            accuracy_threshold (int): The minimum fuzz.ratio() score (0-100) for the
                                    base text (prefix of whisper_text) to be
                                    considered a match against ground_truth. Also used
                                    for fallback overall similarity check.
            min_hallucination_len (int): The minimum number of characters the appended
                                        phrase must have (after stripping) to be
                                        considered a hallucination. Must be > 0.
            base_match_leniency_percent (float): How much shorter or longer the potential base
                                                in whisper_text can be compared to ground_truth,
                                                used to define the search window for the split point.
            hallucination_to_base_max_similarity_threshold (int):
                                    The maximum fuzz.ratio() score (0-100) allowed
                                    between the identified potential base and the identified
                                    potential hallucination. A lower value means the
                                    hallucination must be more dissimilar to the base.

        Returns:
            Tuple[bool, str, str, Optional[int]]:
                - bool: True if hallucination is detected, False otherwise.
                - str: The common "base" text found (preprocessed).
                - str: The "hallucinated" text (preprocessed and stripped).
                - Optional[int]: The similarity score (0-100) between the identified base
                                and the identified hallucination if a hallucination is
                                detected (bool=True and the suffix was checked for dissimilarity).
                                If no hallucination is detected, or if the check wasn't applicable,
                                this is None.
        """
        if min_hallucination_len <= 0:
            raise ValueError("min_hallucination_len must be greater than 0.")

        gt_processed = massage_for_comparison(ground_truth)
        wt_processed = massage_for_comparison(whisper_text)

        if not gt_processed:
            # If ground truth is empty, whisper_text cannot be an "appended" hallucination.
            # If wt_processed is also empty, they match. Otherwise, they don't, but not in the way we test.
            return False, "" if not wt_processed else wt_processed, "", None

        len_gt = len(gt_processed)
        len_wt = len(wt_processed)

        # Early exit: If whisper text isn't long enough to contain ground_truth
        # AND a hallucination of minimum length.
        # Example: GT="abc", min_hallucination_len=5. WT must be at least 3+5=8 chars.
        # If WT="abcdefg" (7 chars), len_wt < len_gt + min_hallucination_len (7 < 3+5) is true.
        if len_wt < len_gt + min_hallucination_len:
            overall_score = fuzz.ratio(gt_processed, wt_processed)
            if overall_score >= accuracy_threshold:
                return False, wt_processed, "", None # Good overall match
            else:
                return False, "", "", None # Different, and too short for this hallucination type

        # Store the best candidate (base + suffix) found *before* final dissimilarity check
        best_potential_candidate = {
            "found": False,
            "base_text": "",
            "suffix_text": "",
            "base_score_to_gt": -1.0, # fuzz.ratio returns float
            "k_len_for_base": -1
        }

        # Determine the search range for k (length of the potential base in whisper_text)
        min_k_search_len = max(1, int(len_gt * (1 - base_match_leniency_percent)) - 2)

        # Max length of base part: must leave at least min_hallucination_len for the suffix.
        max_k_potential_base_len = len_wt - min_hallucination_len

        # Max length of base part: also limited by leniency from gt_len.
        # (e.g., base shouldn't be 2x gt_len and still considered a match for gt)
        max_base_len_from_gt_leniency = int(len_gt * (1 + base_match_leniency_percent)) + 5

        # Effective max k is the minimum of these two constraints on base length
        max_k_search_len = min(max_k_potential_base_len, max_base_len_from_gt_leniency)

        if max_k_search_len < min_k_search_len:
            # Constraints (leniency, min_hallucination_len) are too tight for given string lengths.
            overall_score = fuzz.ratio(gt_processed, wt_processed)
            if overall_score >= accuracy_threshold:
                return False, wt_processed, "", None
            return False, "", "", None

        for k_candidate_len in range(min_k_search_len, max_k_search_len + 1):
            if k_candidate_len <= 0 : # Should be covered by min_k_search_len >= 1, but defensive.
                continue

            current_base_candidate = wt_processed[:k_candidate_len]
            current_suffix_candidate = wt_processed[k_candidate_len:].strip() # Strip suffix here

            # Condition 1: Base candidate must match ground_truth sufficiently
            base_to_gt_score = fuzz.ratio(gt_processed, current_base_candidate)
            if base_to_gt_score < accuracy_threshold:
                continue

            # Condition 2: Suffix candidate must meet minimum length (after stripping)
            if len(current_suffix_candidate) < min_hallucination_len:
                continue

            # This k_candidate_len results in a potential base and a sufficiently long suffix.
            # Is it better than what we've found so far?
            is_better = False
            if not best_potential_candidate["found"]:
                is_better = True
            elif base_to_gt_score > best_potential_candidate["base_score_to_gt"]:
                is_better = True
            elif base_to_gt_score == best_potential_candidate["base_score_to_gt"]:
                current_k_diff = abs(k_candidate_len - len_gt)
                best_k_diff = abs(best_potential_candidate["k_len_for_base"] - len_gt)
                if current_k_diff < best_k_diff:
                    is_better = True
                elif current_k_diff == best_k_diff and k_candidate_len > best_potential_candidate["k_len_for_base"]:
                    is_better = True # Prefer longer base if diffs and scores are identical

            if is_better:
                best_potential_candidate["found"] = True
                best_potential_candidate["base_text"] = current_base_candidate
                best_potential_candidate["suffix_text"] = current_suffix_candidate
                best_potential_candidate["base_score_to_gt"] = base_to_gt_score
                best_potential_candidate["k_len_for_base"] = k_candidate_len

        # After checking all split points:
        if best_potential_candidate["found"]:
            potential_base = best_potential_candidate["base_text"]
            potential_suffix = best_potential_candidate["suffix_text"]

            # Condition 3: The identified suffix must be dissimilar to the identified base
            similarity_suffix_to_base = fuzz.ratio(potential_base, potential_suffix)
            similarity_score_int = int(round(similarity_suffix_to_base))

            if similarity_suffix_to_base < hallucination_to_base_max_similarity_threshold:
                # All conditions met for this specific type of hallucination
                return True, potential_base, potential_suffix, similarity_score_int
            # else: The suffix was too similar to the base. This is not the "random appended" type.
            # Fall through to overall similarity check, returning the (high) similarity score for context.
            # However, the primary return bool should be False for "hallucination detected".
            # If we return the similarity_score_int here, it might be confusing.
            # The request was to return similarity "if a hallucination is detected".
            # So if it's *not* detected due to this check, the similarity is None.

        # If no potential candidate was found (due to base match or suffix length),
        # OR if the best one had a suffix too similar to its base:
        # Fallback to checking overall similarity between ground_truth and whisper_text.
        overall_score = fuzz.ratio(gt_processed, wt_processed)
        if overall_score >= accuracy_threshold:
            # Texts are similar enough overall; consider wt_processed as the "base"
            # and no appended hallucination of the specific type.
            return False, wt_processed, "", None
        else:
            # Texts are different, and no specific appended hallucination was found.
            return False, "", "", None

    @staticmethod
    def make_items(state: State) -> list[Item]:
        """
        Creates items, including their audio file duration values
        """
        items = []
        index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        for index, file_path in index_to_path.items():
            text_segment = state.text_segments[index]
            seconds = AppUtil.get_flac_file_duration(file_path)
            if not isinstance(seconds, float):
                L.w(f"Ignoring - couldn't get duration for {file_path}")
                continue
            items.append( Item(index, file_path, text_segment, seconds) )
        return items

    @staticmethod
    def populate_transcribed_text(item: Item, whisper_model) -> None:
        result = whisper_model.transcribe(item.path, language=None)
        if not result or not "text" in result:
            item.transcribed_text = TranscribeFail()
        else:
            text = result["text"]
            if not isinstance(text, str):
                L.w(f"unexpected whisper result: {text}")
                item.transcribed_text = TranscribeFail()
            else:
                item.transcribed_text = text.strip()

    @staticmethod
    def populate_transcribed_text_all(items: list[Item]) -> None:

        printt("Initializing whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("turbo", device=device)
        printt()

        for i, item in enumerate(items):
            BadGenUtil.populate_transcribed_text(item, whisper_model)
            if i % 25 == 25:
                print(i)

        # Cleanup
        del whisper_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def print_item_fail(item: Item) -> None:
        if not isinstance(item.result, FailResult):
            return
        prefix = f"{make_hotkey_string(str(item.index))}"
        len_prefix = len(strip_ansi_codes(prefix))
        printt(f"{prefix} {item.path}")
        print(f"{" " * len_prefix} {item.result.message}")
        print()

# ---

def massage_for_comparison(s: str) -> str:
    # Massages text so that source text can be reliably compared to transcribed text
    s = s.lower().strip()
    # First replace fancy apost with normal apost
    s = s.replace("’", "'") #
    # Replace all non-alpha-numerics with space, except for apost that is inside a word
    s = re.sub(r"[^a-zA-Z0-9'’]|(?<![a-zA-Z])['’]|['’](?![a-zA-Z])", ' ', s)
    s = re.sub(r' +', ' ', s)
    s = s.strip(' ')
    return s

def rename_as_passed(file_path_string: str) -> None:
    """ Inserts "[pass]" into the file name at end of 'bracket list' """
    try:
        path = Path(file_path_string)
        if not path.is_file():
            L.w(f"File not found, cannot rename: {file_path_string}")
            return

        directory = path.parent
        stem = path.stem
        suffix = path.suffix
        token_to_insert = " [pass]"

        # Find all non-overlapping occurrences of bracketed segments like "[text]"
        matches = list(re.finditer(r"\[[^\]]+\]", stem))

        new_stem: str
        if len(matches) >= 3:
            # Insert after the third bracketed segment
            insert_pos = matches[2].end()
            new_stem = stem[:insert_pos] + token_to_insert + stem[insert_pos:]
        else:
            # Append to the end of the stem
            new_stem = stem + token_to_insert

        new_name = new_stem + suffix
        new_path = directory / new_name

        if path == new_path:
            L.i(f"File {file_path_string} name is already in the desired format or no change needed.")
            return

        path.rename(new_path)

    except Exception as e:
        L.w(f"Error renaming file '{file_path_string}': {type(e).__name__} - {e}")

def rename_dev(filepath, replacement_pattern="bad"):
    # Split into directory and filename components
    path_obj = Path(filepath)
    directory = str(path_obj.parent)
    filename = path_obj.name

    # Process just the filename part
    parts = re.findall(r'\[.*?\]|.*?(?=\[|$)', filename)
    bracketed = [part for part in parts if part.startswith('[') and part.endswith(']')]

    if len(bracketed) >= 3:
        random_num = random.randint(1000, 9999)
        new_segment = f"[{replacement_pattern}_{random_num}]"
        bracketed[2] = new_segment

        new_parts = []
        bracket_index = 0
        for part in parts:
            if part.startswith('[') and part.endswith(']'):
                new_parts.append(bracketed[bracket_index])
                bracket_index += 1
            else:
                new_parts.append(part)

        new_filename = ''.join(new_parts)
        # Reconstruct full path with original directory
        return os.path.join(directory, new_filename)
    else:
        return filepath  # return original if fewer than 3 bracketed segments

# ---

class Item:
    def __init__(self, i: int, path: str, text: str, duration: float):
        self.index = i
        self.path = path
        self.text = text
        self.duration = duration
        self.transcribed_text: str | TranscribeFail | Untranscribed = Untranscribed()
        self.result: PassResult | FailResult | None = None

class TranscribeFail:
    pass

class Untranscribed:
    pass

class PassResult:
    pass

class FailResult:
    def __init__(self, message: str):
        self.message = message