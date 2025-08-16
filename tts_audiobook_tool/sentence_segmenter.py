from __future__ import annotations
import re

class SentenceSegmenter:

    # comma, semicolon, colon, en-dash, em-dash, open paren, close paren
    # not including normal dash (which can just as often be used as a connecting-character-between-words)
    PHRASE_DELIMITERS = r'[(),;:\–\—]'

    @staticmethod
    def segment_sentence(sentence, max_words=25):
        """
        Splits a single sentence into smaller segments (phrases) based on
        word count limits and preferred split points (commas, spaces near middle),
        preserving internal whitespace and splitting characters.

        Args:
            sentence (str): The input sentence (assumed to be a single sentence).
            max_words (int): The maximum number of words allowed in each segment.

        Returns:
            list[str]: A list of phrase strings, preserving whitespace.
        """

        # Keep original sentence with its whitespace
        if not sentence: # Only check for empty/None, not whitespace content
            return []
        # Return original sentence if max_words is invalid, keeping whitespace
        if max_words <= 0:
            return [sentence]

        result_phrases = []
        current_segment = sentence # Start with the original sentence, no stripping

        # Keep track of the original word count for the safety check
        original_word_count_for_loop = len(current_segment.split()) if current_segment else 0

        while current_segment:
            # Use split() only for counting, not for altering the segment
            word_count = len(current_segment.split())

            if word_count <= max_words:
                # Add the final remaining segment with its whitespace
                result_phrases.append(current_segment)
                break # Finished processing
            else:
                # Segment is too long, find where to split it
                # _find_best_split_point now returns the index *after* the end of the first segment
                split_end_index = SentenceSegmenter._find_best_split_point(current_segment, max_words)

                if split_end_index is None or split_end_index <= 0:
                     # Safety break 1: Couldn't find a valid split point at all.
                     # Add the whole remaining segment and stop.
                    print(f"Warning: Could not find any valid split point for segment: '{current_segment[:50]}...'")
                    result_phrases.append(current_segment)
                    break
                elif split_end_index >= len(current_segment):
                     # Safety break 2: Split point is at or after the end (e.g., fallback error).
                     # Add the whole remaining segment and stop.
                    print(f"Warning: Calculated split point is at/after end for segment: '{current_segment[:50]}...'")
                    result_phrases.append(current_segment)
                    break


                # Extract the phrase *including* the splitting character/whitespace up to split_end_index
                phrase = current_segment[:split_end_index]
                if phrase: # Avoid adding empty strings if split is at the very beginning
                     result_phrases.append(phrase)

                # Update the segment to the remaining part, starting right after the first part
                current_segment = current_segment[split_end_index:]

                # Safety check for infinite loops: if the new segment is identical or longer in words
                new_word_count = len(current_segment.split()) if current_segment else 0
                if new_word_count >= word_count:
                    print(f"Warning: Split did not reduce word count or segment unchanged, stopping. Segment: '{current_segment[:50]}...'")
                    # Add the problematic remainder to avoid losing text and stop
                    if current_segment:
                        result_phrases.append(current_segment)
                    break
                # Update word_count for the next iteration's comparison in the loop check itself
                # This is redundant now that we check inside, but doesn't hurt.
                # word_count = new_word_count


        # Filter out potentially empty strings if splitting resulted in them (e.g., splitting on multiple spaces)
        # Though the logic should generally avoid this now. Keep it as a safeguard.
        return [p for p in result_phrases if p]

    @staticmethod
    def _find_best_split_point(segment, max_words):
        """
        Finds the best character index to split a segment that is too long.
        The returned index marks the position *after* the end of the first segment.
        Prioritizes commas/colons/semicolons near the middle, then spaces near the middle,
        then falls back to the space after max_words.
        Returns the character index for slicing (exclusive end for first part).
        """
        words = segment.split() # Simple split for counting words
        word_count = len(words)

        if word_count <= max_words:
            return None # No split needed based on length

        # --- Define search center and range ---
        middle_char_index = len(segment) // 2
        search_radius = max(20, len(segment) // 4)

        # --- 1. Look for punc (commas/semicolons/colons) near the middle ---
        best_punc_split_idx = -1
        min_punc_dist = float('inf')

        start_search = max(0, middle_char_index - search_radius)
        end_search = min(len(segment), middle_char_index + search_radius)

        # Find all punc in the search range
        punc_indices = [m.start() for m in re.finditer(SentenceSegmenter.PHRASE_DELIMITERS, segment[start_search:end_search])]

        punc_indices = [idx + start_search for idx in punc_indices]

        for idx in punc_indices:
            # Potential split point is *after* the punc.
            split_point = idx + 1
            # Check word count constraint for the left part (segment[:split_point])
            left_part = segment[:split_point]
            if len(left_part.split()) <= max_words:
                dist = abs(split_point - middle_char_index)
                if dist < min_punc_dist:
                    min_punc_dist = dist
                    best_punc_split_idx = split_point # Store index *after* punc

        if best_punc_split_idx != -1:
            # We found a punc split. Return the index right after the punc.
            # The space(s) after the punc will start the next segment.
            # print(f"Debug: Found punc split point (index after punc): {best_punc_split_idx}")
            return best_punc_split_idx

        # --- 2. Look for spaces near the middle ---
        best_space_split_idx = -1
        min_space_dist = float('inf')

        # Find all spans of whitespace in the search range
        space_matches = list(re.finditer(r'\s+', segment[start_search:end_search]))

        for match in space_matches:
            # Potential split point is *after* the whitespace span
            split_point = match.end() + start_search # Adjust index to full segment
            # Check word count constraint for the left part (segment[:split_point])
            left_part = segment[:split_point]
            # Ensure the split doesn't happen right at the beginning
            if split_point > 0 and len(left_part.split()) <= max_words:
                dist = abs(split_point - middle_char_index)
                if dist < min_space_dist:
                    min_space_dist = dist
                    best_space_split_idx = split_point # Store index *after* the space

        if best_space_split_idx != -1:
             # We found a space split. Return the index right after the space(s).
             # print(f"Debug: Found space split point near middle (index after space): {best_space_split_idx}")
            return best_space_split_idx

        # --- 3. Fallback: Split strictly after max_words ---
        # Find the character index *after* the end of the max_words-th word.
        # This means finding the start of the (max_words + 1)-th word.
        fallback_split_idx = SentenceSegmenter._find_split_char_index(segment, max_words + 1)

        if fallback_split_idx is not None and fallback_split_idx != -1 and fallback_split_idx > 0:
            # fallback_split_idx is the start of the next word. This is where the next segment begins.
            # So, the first segment ends just before this index.
            # print(f"Debug: Using fallback split point (start of word {max_words + 1}): {fallback_split_idx}")
            return fallback_split_idx
        else:
            # Error case: Couldn't find the start of the (max_words + 1)-th word.
            # This might mean the segment has <= max_words words already (handled earlier),
            # or it's one giant word with no spaces, or an issue with _find_split_char_index.
            # Return None to indicate failure to find a split point here.
            print(f"Warning: Could not find a valid fallback split point for segment exceeding {max_words} words. Segment: '{segment[:50]}...'")
            return None # Signal failure to the caller

    @staticmethod
    def _find_split_char_index(text, target_word_index):
        """
        Finds the starting character index in the original text for the word
        at the 1-based target_word_index. Handles multiple spaces.
        Returns -1 if index is out of bounds or word not found.
        """
        word_count = 0
        in_word = False
        start_index = -1

        for i, char in enumerate(text):
            is_space = char.isspace()

            if not is_space and not in_word:
                # Start of a new word
                in_word = True
                word_count += 1
                if word_count == target_word_index:
                    start_index = i
                    break # Found the start of the target word
            elif is_space:
                # End of a potential word
                in_word = False

        # If loop finishes without finding the start_index (e.g., target_word_index > actual words)
        # start_index remains -1.
        return start_index


# ---



if __name__ == "__main__":
    test_cases = [
        # Basic Tests
        {
            "description": "Basic split needed",
            "input_sentence": "This is a slightly longer test sentence that needs to be split.",
            "max_words": 5,
            "expected_output": [
                "This is a slightly longer ",
                "test sentence that needs ",
                "to be split."
            ]
        },
        {
            "description": "Sentence fits within limit",
            "input_sentence": "This sentence is short enough.",
            "max_words": 10,
            "expected_output": ["This sentence is short enough."]
        },
        {
            "description": "Sentence length equals limit",
            "input_sentence": "One two three four five.",
            "max_words": 5,
            "expected_output": ["One two three four five."]
        },
        # Edge Cases: Input
        {
            "description": "Empty string input",
            "input_sentence": "",
            "max_words": 5,
            "expected_output": []
        },
        {
            "description": "None input",
            "input_sentence": None,
            "max_words": 5,
            "expected_output": []
        },
        {
            "description": "Whitespace only input",
            "input_sentence": "   \t \n ",
            "max_words": 5,
            "expected_output": ["   \t \n "] # Contains only whitespace, word count is 0 <= 5
        },
        # Edge Cases: max_words
        {
            "description": "max_words = 0",
            "input_sentence": "Test with max_words zero.",
            "max_words": 0,
            "expected_output": ["Test with max_words zero."]
        },
        {
            "description": "max_words < 0",
            "input_sentence": "Test with negative max_words.",
            "max_words": -5,
            "expected_output": ["Test with negative max_words."]
        },
         {
            "description": "max_words = 1",
            "input_sentence": "Split every single word.",
            "max_words": 1,
            "expected_output": ["Split ", "every ", "single ", "word."]
        },
        # Whitespace Preservation
        {
            "description": "Multiple spaces between words",
            "input_sentence": "Sentence  with   extra    spaces.",
            "max_words": 2,
            "expected_output": ["Sentence  with   ", "extra    spaces."]
        },
        {
            "description": "Leading and trailing spaces",
            "input_sentence": "  Sentence has leading and trailing spaces.  ",
            "max_words": 4,
            "expected_output": ["  Sentence has leading and ", "trailing spaces.  "]
        },
        {
            "description": "Tab characters",
            "input_sentence": "Split\twith\ttabs\tinstead.",
            "max_words": 2,
            "expected_output": ["Split\twith\t", "tabs\tinstead."]
        },
         {
            "description": "Mixed whitespace",
            "input_sentence": "  Leading space, then\t tab and  multiple spaces.",
            "max_words": 3,
            "expected_output": ["  Leading space, then\t ", "tab and  multiple ", "spaces."] # Comma split first
        },
        # Punctuation Splitting
        {
            "description": "Prioritize comma near middle",
            "input_sentence": "This is the first part, and this is the significantly longer second part.",
            "max_words": 6, # Middle is around 'significantly'
            "expected_output": ["This is the first part, ", "and this is the significantly ", "longer second part."] # Split at comma first
        },
        {
            "description": "Prioritize semicolon near middle",
            "input_sentence": "Clause one is shorter; clause two rambles on for quite a while longer.",
            "max_words": 6, # Middle is around 'rambles'
            "expected_output": ["Clause one is shorter; ", "clause two rambles on for ", "quite a while longer."] # Split at semicolon first
        },
        {
            "description": "Prioritize colon near middle",
            "input_sentence": "Here is the introduction: the main point follows and is quite detailed.",
            "max_words": 5, # Middle is around 'point'
            "expected_output": ["Here is the introduction: ", "the main point follows and ", "is quite detailed."] # Split at colon first
        },
        {
            "description": "Punctuation with extra space",
            "input_sentence": "Split here,   then continue with the rest.",
            "max_words": 3,
            "expected_output": ["Split here,   ", "then continue with ", "the rest."] # Split includes space after comma
        },
        {
            "description": "Multiple punctuation options, choose middle",
            "input_sentence": "First part is short, second part is medium length; the third part is the longest of all.",
            "max_words": 7, # Middle around "; the third"
            "expected_output": ["First part is short, second part is medium length; ", "the third part is the longest of all."] # Semicolon is closer to middle
        },
        # Space Splitting (Middle Priority)
        {
            "description": "No punctuation, split at space near middle",
            "input_sentence": "This sentence has no punctuation marks but is long enough to require splitting somewhere near the middle.",
            "max_words": 8, # Middle around 'require splitting'
            "expected_output": [
                "This sentence has no punctuation marks but ", # Space split near middle preferred
                "is long enough to require splitting somewhere ",
                "near the middle."
            ]
        },
        # Fallback Splitting
        {
            "description": "Fallback to max_words limit (no good middle split)",
            "input_sentence": "A very simple and straightforward sentence structure for testing the fallback split.",
             # Max 6 words. No obvious middle punc/space. Fallback after 'straightforward '.
            "max_words": 6,
            "expected_output": [
                "A very simple and straightforward sentence ",
                "structure for testing the fallback split."
             ]
        },
        # Complex Cases
        {
            "description": "Long word (cannot split within)",
            "input_sentence": "Features Pneumonoultramicroscopicsilicovolcanoconiosis which is long.",
            "max_words": 2,
            "expected_output": [
                "Features ",
                "Pneumonoultramicroscopicsilicovolcanoconiosis ", # This segment has 1 word > max_words, but can't be split
                "which is long."
            ]
        },
        {
            "description": "No spaces at all (cannot split)",
            "input_sentence": "Thisissentencewithnospacesinit.",
            "max_words": 5,
            "expected_output": ["Thisissentencewithnospacesinit."] # Cannot split, returns original
        },
         {
            "description": "Mix of strategies needed",
            "input_sentence": "Start here, then continue for a while; after the semicolon we need another split possibly using the fallback.",
            "max_words": 5,
             "expected_output": [
                "Start here, ",                           # Punc split
                "then continue for a while; ",            # Punc split
                "after the semicolon we need ",           # Fallback split (after 'need ')
                "another split possibly using the ",      # Fallback split (after 'the ')
                "fallback."
             ]
         },
         {
            "description": "Splitting near start due to low max_words",
            "input_sentence": "This very long sentence needs splitting early.",
            "max_words": 2,
            "expected_output": [
                "This very ",
                "long sentence ",
                "needs splitting ",
                "early."
            ]
         }
    ]

    passed_count = 0
    failed_count = 0

    print("--- Running SentenceSegmenter Tests ---")

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['description']}")
        print(f"Input: '{test['input_sentence']}' (max_words={test['max_words']})")
        try:
            actual_output = SentenceSegmenter.segment_sentence(test['input_sentence'], test['max_words'])
            expected_output = test['expected_output']

            if actual_output == expected_output:
                print(f"Status: PASSED")
                passed_count += 1
            else:
                print(f"Status: FAILED")
                print(f"  Expected: {expected_output}")
                print(f"  Actual:   {actual_output}")
                failed_count += 1
        except Exception as e:
            print(f"Status: ERROR")
            print(f"  Exception: {e}")
            failed_count += 1


    print("\n--- Test Summary ---")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("--- End of Tests ---")