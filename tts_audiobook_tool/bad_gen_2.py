from typing import Optional, Tuple
from tts_audiobook_tool.util import *
from rapidfuzz import fuzz

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

    gt_processed = massage_for_text_comparison(ground_truth)
    wt_processed = massage_for_text_comparison(whisper_text)

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



            # if fail_reason:
            #     # Whisper can hallucinate extra words at the end and create false positives
            #     # ("condition_on_previous_text=False" did not change this one way or another, even though it's supposed to.)
            #     # But outte has a tendency to hallucinate too, where it will repeat phrases or sentences at the end as well.
            #     # So these two issues overlap.
            #     # However the outte hallucinations repeat previous phrases but whisper ones append short unrelated ones, so.
            #     # It's much more important to under-detect false positives rather than the opposite, since being forced to regenerate
            #     # false positives is relatively harmess, versus letting a bad generation go through, which is much worse.
            #     if False:
            #         # Work in progress, meh
            #         tup = BadGenUtil.detect_whisper_hallucination(massaged_text, massaged_transcribed_text)
            #         if tup[0] == True:
            #             fail_reason = ""
            #             print("fail reason rescinded:", tup)

