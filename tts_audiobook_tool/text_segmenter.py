import pysbd

from tts_audiobook_tool.sentence_segmenter import SentenceSegmenter
from tts_audiobook_tool.text_segment import TextSegment, TextSegmentReason

class TextSegmenter:

    @staticmethod
    def segment_text(full_text: str, max_words: int, language="en") -> list[TextSegment]:
        """
        Segments full project text into chunks.
        Segments by sentence. When sentence is longer than max_words, splits sentence into chunks.
        """

        # Pass 1: Segment text into sentences using pysbd
        # Important: "clean=False" preserves leading and trailing whitespace
        segmenter = pysbd.Segmenter(language=language, clean=False, char_span=False)
        texts = segmenter.segment(full_text)

        # Pass 2: pysbd treats everything enclosed in quotes as a single sentence, so split those up
        new_texts = []
        for text in texts:
            if starts_and_ends_with_quote(text):
                lst = segment_quote_text(text, segmenter)
                new_texts.extend(lst)
            else:
                new_texts.append(text)
        texts = new_texts

        # Pass 3: Split any segments that are longer than max_words using own algo
        tups = []
        for text in texts:
            lst = SentenceSegmenter.segment_sentence(text, max_words=max_words)
            for i, subsegment in enumerate(lst):
                reason = TextSegmentReason.SENTENCE if i == 0 else TextSegmentReason.INSIDE_SENTENCE
                tups.append( (subsegment, reason) )

        # Make TextSegments proper
        counter = 0
        text_segments: list[TextSegment] = []
        for text, reason in tups:
            length = len(text)
            text_segment = TextSegment(
                text=text, index_start=counter, index_end=counter + length, reason=reason
            )
            text_segments.append(text_segment)
            counter += length

        # Pass 4 Set paragraph breaks
        for i in range(1, len(text_segments)):
            segment_a = text_segments[i - 1]
            segment_b = text_segments[i]
            if has_trailing_line_break(segment_a.text):
                segment_b.reason = TextSegmentReason.PARAGRAPH

        # Pass 5 - merge short segments
        text_segments = merge_short_segments_all(text_segments, max_words)

        return text_segments

# ---

def merge_short_segments_all(segments: list[TextSegment], max_words: int) -> list[TextSegment]:

    result = []

    # Merge only within paragraphs
    # TODO Reconsider that. Chatterbox (and probably oute) fails _a lot_ on one-word gens.
    paragraphs = make_paragraph_lists(segments)

    for paragraph in paragraphs:
        items = merge_short_segments(paragraph, max_words)
        result.extend(items)

    return result

def merge_short_segments(segments: list[TextSegment], max_words: int) -> list[TextSegment]:
    """
    Merges short segments with their neighbors.
    A merge is only performed if the combined word count does not exceed max_words.
    If segment starts with a quote, dont' merge with previous.
    If segment ends with quote, don't merge with next
    """
    if not segments:
        return []

    merged_segments = list(segments)

    while True:

        was_merged_in_pass = False
        i = 0

        while i < len(merged_segments):

            current_segment = merged_segments[i]

            # Two words seems to do mostly alright (Chatterbox and Oute).
            # One word definitely does not.
            MAX_WORDS = 1
            if word_count(current_segment.text) > MAX_WORDS:
                i += 1
                continue

            # Current segment is short, and is not a self-contained quote.

            # Try to merge with the next segment
            if i + 1 < len(merged_segments):

                next_segment = merged_segments[i+1]
                merged_text = current_segment.text + next_segment.text

                should = word_count(merged_text) <= max_words
                if should:
                    # Perform the merge
                    new_segment = TextSegment(
                        text=merged_text,
                        index_start=current_segment.index_start,
                        index_end=next_segment.index_end,
                        reason=current_segment.reason
                    )
                    merged_segments[i] = new_segment
                    del merged_segments[i+1]
                    was_merged_in_pass = True
                    # A merge happened, so we restart the scan to re-evaluate from the beginning
                    break

            # Try to merge with the previous segment
            if i > 0:

                prev_segment = merged_segments[i-1]
                merged_text = prev_segment.text + current_segment.text

                should = word_count(merged_text) <= max_words
                if should:
                    # Perform the merge
                    new_segment = TextSegment(
                        text=merged_text,
                        index_start=prev_segment.index_start,
                        index_end=current_segment.index_end,
                        reason=prev_segment.reason
                    )
                    merged_segments[i-1] = new_segment
                    del merged_segments[i]
                    was_merged_in_pass = True
                    # A merge happened, so we restart the scan
                    break

            i += 1

        # If we went through a whole pass without any merges, we're done.
        if not was_merged_in_pass:
            break

    return merged_segments


def make_paragraph_lists(segments: list[TextSegment]) -> list[list[TextSegment]]:

    paragraphs = []

    paragraph = []
    for i, segment in enumerate(segments):
        if segment.reason == TextSegmentReason.PARAGRAPH:
            paragraphs.append(paragraph)
            paragraph = []
        paragraph.append(segment)

    if paragraph:
        paragraphs.append(paragraph)

    return paragraphs


def segment_quote_text(text: str, segmenter) -> list[str]:
    """
    Given a quote which may consist of multiple sentences and may have whitespace before and/or after the quote,
    segment the quote by sentence, preserving whitespace.
    """
    before, content, after = split_string_parts(text)
    segments = segmenter.segment(content)
    segments[0] = before + segments[0]
    segments[-1] = segments[-1] + after
    return segments

def split_string_parts(text: str) -> tuple[str, str, str]:
    """
    Splits a string into three parts:
    - before: Leading whitespace + first non-whitespace character
    - content: Everything between before and end
    - after: Last non-whitespace character + trailing whitespace
    Returns:
        Tuple of (before, content, end)
    """
    if not text:
        return ('', '', '')

    # Get leading whitespace
    leading_ws = text[:len(text) - len(text.lstrip())]

    # Calculate 'before'
    stripped_left = text.lstrip()
    before = leading_ws + (stripped_left[0] if stripped_left else '')

    # Calculate 'end'
    stripped_right = text.rstrip()
    trailing_ws = text[len(stripped_right):] if stripped_right else text
    after = (stripped_right[-1] if stripped_right else '') + trailing_ws

    # Calculate 'content'
    content_start = len(before)
    content_end = -len(after) if after else None
    content = text[content_start:content_end]

    return (before, content, after)

def starts_and_ends_with_quote(s: str) -> bool:
    start, _, end = split_string_parts(s)
    return has_quote_char(start) and has_quote_char(end)

def has_quote_char(s: str) -> bool:
    for char in QUOTATION_CHARS:
        if char in s:
            return True
    return False

def has_trailing_line_break(s: str) -> bool:
    trailing_whitespace = s[len(s.rstrip()):]
    return "\n" in trailing_whitespace

def word_count(text: str) -> int:
    """Counts words in a string. Strips leading/trailing whitespace and splits."""
    return len(text.strip().split())


QUOTATION_CHARS = "\"'“”"