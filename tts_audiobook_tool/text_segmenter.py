import pysbd

from tts_audiobook_tool.sentence_segmenter import SentenceSegmenter
from tts_audiobook_tool.text_segment import TextSegment, TextSegmentReason
from tts_audiobook_tool.text_util import TextUtil

class TextSegmenter:

    @staticmethod
    def segment_text(source_text: str, max_words: int, language="en") -> list[TextSegment]:
        """
        Splits source text into "TextSegment" chunks.
        App's main text segmentation algorithm.
        """

        # Pass 1: Segment text into sentences using pysbd
        # Important: "clean=False" preserves leading and trailing whitespace
        segmenter = pysbd.Segmenter(language=language, clean=False, char_span=False)
        texts = segmenter.segment(source_text)

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
                if i == 0:
                    reason = TextSegmentReason.SENTENCE
                else:
                    previous_subsegment = lst[i - 1].strip()
                    last_char = previous_subsegment[-1]
                    if last_char.isalpha():
                        reason = TextSegmentReason.WORD
                    else:
                        reason = TextSegmentReason.PHRASE # assumption, good enough
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

        # Pass 4 Set paragraph break reason
        for i in range(1, len(text_segments)):
            segment_a = text_segments[i - 1]
            segment_b = text_segments[i]
            count = num_trailing_line_breaks(segment_a.text)
            if 1 <= count <= 2:
                segment_b.reason = TextSegmentReason.PARAGRAPH
            elif count > 2:
                segment_b.reason = TextSegmentReason.SECTION

        # Pass 5 - merge short segments
        text_segments = merge_short_segments_all(text_segments, max_words)

        # Pass 6 - Filter out items w/o 'vocalizable' content
        # TODO: all text should be retained for display purposes. wd require a lot of reworking tho.
        text_segments = [item for item in text_segments if has_alpha_numeric_char(item.text)]
        return text_segments

    @staticmethod
    def segment_text_paragraphs(full_text: str) -> list[TextSegment]:
        """
        Segments source text into paragraphs.
        Currently unused
        """

        lines = TextUtil.split_text_into_paragraphs(full_text)

        text_segments = []
        counter = 0
        for line in lines:
            text_segment = TextSegment(line, counter, counter + len(line), TextSegmentReason.PARAGRAPH)
            counter += len(line)
            text_segments.append(text_segment)

        return text_segments

# ---

def merge_short_segments_all(segments: list[TextSegment], max_words: int) -> list[TextSegment]:
    result = []
    # Will merge only within a paragraph
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

            # Two words prompts only occassionally has problems (Chatterbox and Oute).
            # Single word often does.
            MAX_WORDS = 2
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
    """
    Groups text segments by paragraph/section
    """
    paragraphs = []

    paragraph = []
    for i, segment in enumerate(segments):
        if segment.reason in [TextSegmentReason.PARAGRAPH, TextSegmentReason.SECTION]:
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

def starts_and_ends_with_quote(s: str) -> bool:
    start, _, end = split_string_parts(s)
    return has_quote_char(start) and has_quote_char(end)

def split_string_parts(text: str) -> tuple[str, str, str]:
    """
    Splits a string into three parts:
    - before: Leading whitespace + first non-whitespace character
    - content: Everything between before and after
    - after: Last non-whitespace character + trailing whitespace

    If there is only one non-whitespace character,
    it should be assigned to "before", and "content" should be empty.

    If there are only two non-whitespace characters,
    the first character should be assigned to "before",
    the second character should be assigned to "after",
    and "content" should be empty.

    Returns:
        Tuple of (before, content, after)
    """
    if not text:
        return ('', '', '')

    stripped_text = text.strip()

    # Handle strings that are empty or contain only whitespace.
    # In this case, 'before' contains the whole string.
    if not stripped_text:
        return (text, '', '')

    # Find the indices of the first and last non-whitespace characters.
    # This is a more direct and robust way to find the split points.
    first_char_index = text.find(stripped_text[0])
    last_char_index = text.rfind(stripped_text[-1])

    # If the first and last non-whitespace character is the same,
    # it fully belongs to 'before' as per the docstring.
    if first_char_index == last_char_index:
        before = text[:first_char_index + 1]
        content = ''
        after = text[first_char_index + 1:]
        return (before, content, after)

    # For all other cases (2 or more non-whitespace characters),
    # slice the string based on the found indices.
    before = text[:first_char_index + 1]
    content = text[first_char_index + 1:last_char_index]
    after = text[last_char_index:]

    return (before, content, after)

def has_quote_char(s: str) -> bool:
    for char in QUOTATION_CHARS:
        if char in s:
            return True
    return False

def num_trailing_line_breaks(s: str) -> int:
    trailing_whitespace = s[len(s.rstrip()):]
    return trailing_whitespace.count("\n")

def word_count(text: str) -> int:
    """Counts words in a string. Strips leading/trailing whitespace and splits."""
    return len(text.strip().split())

def has_alpha_numeric_char(s: str) -> bool:
    return any(c.isalnum() for c in s)


QUOTATION_CHARS = "\"'‘’“”"

#  ' ), the closing single quote ( ' )
