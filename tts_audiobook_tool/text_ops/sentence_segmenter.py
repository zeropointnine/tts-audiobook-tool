from __future__ import annotations
import re

from tts_audiobook_tool.util import printt

class SentenceSegmenter:

    @staticmethod
    def segment_sentence(sentence, max_words:int):
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
                    printt(f"Warning: Could not find any valid split point for segment: '{current_segment[:50]}...'")
                    result_phrases.append(current_segment)
                    break
                elif split_end_index >= len(current_segment):
                     # Safety break 2: Split point is at or after the end (e.g., fallback error).
                     # Add the whole remaining segment and stop.
                    printt(f"Warning: Calculated split point is at/after end for segment: '{current_segment[:50]}...'")
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
                    printt(f"Warning: Split did not reduce word count or segment unchanged, stopping. Segment: '{current_segment[:50]}...'")
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
        punc_indices = [m.start() for m in re.finditer(PHRASE_DELIMITERS, segment[start_search:end_search])]

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
            # printt(f"Debug: Found punc split point (index after punc): {best_punc_split_idx}")
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
             # printt(f"Debug: Found space split point near middle (index after space): {best_space_split_idx}")
            return best_space_split_idx

        # --- 3. Fallback: Split strictly after max_words ---
        # Find the character index *after* the end of the max_words-th word.
        # This means finding the start of the (max_words + 1)-th word.
        fallback_split_idx = SentenceSegmenter._find_split_char_index(segment, max_words + 1)

        if fallback_split_idx is not None and fallback_split_idx != -1 and fallback_split_idx > 0:
            # fallback_split_idx is the start of the next word. This is where the next segment begins.
            # So, the first segment ends just before this index.
            # printt(f"Debug: Using fallback split point (start of word {max_words + 1}): {fallback_split_idx}")
            return fallback_split_idx
        else:
            # Error case: Couldn't find the start of the (max_words + 1)-th word.
            # This might mean the segment has <= max_words words already (handled earlier),
            # or it's one giant word with no spaces, or an issue with _find_split_char_index.
            # Return None to indicate failure to find a split point here.
            printt(f"Warning: Could not find a valid fallback split point for segment exceeding {max_words} words. Segment: '{segment[:50]}...'")
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

# comma, semicolon, colon, en-dash, em-dash, open paren, close paren
# not including normal dash (which can just as often be used as a connecting-character-between-words)
PHRASE_DELIMITERS = r'[(),;:\–\—]'

