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
            if is_quotation(text):
                lst = segment_quote(text, segmenter)
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

        # Pass 4 retroactive - assign paragraph reason
        for i in range(1, len(text_segments)):
            segment_a = text_segments[i - 1]
            segment_b = text_segments[i]
            if has_trailing_line_break(segment_a.text):
                segment_b.reason = TextSegmentReason.PARAGRAPH

        return text_segments

# ---

def has_trailing_line_break(s: str) -> bool:
    trailing_whitespace = s[len(s.rstrip()):]
    return "\n" in trailing_whitespace

def is_quotation(s: str) -> bool:
    """
    Returns True if stripped string starts and ends with quotation characters
    """
    s = s.strip()
    if len(s) <= 3:
        return False
    first = s[0]
    last = s[-1]
    return first in QUOTATION_CHARS and last in QUOTATION_CHARS


def segment_quote(text: str, segmenter) -> list[str]:
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
    - end: Last non-whitespace character + trailing whitespace
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
    end = (stripped_right[-1] if stripped_right else '') + trailing_ws

    # Calculate 'content'
    content_start = len(before)
    content_end = -len(end) if end else None
    content = text[content_start:content_end]

    return (before, content, end)

QUOTATION_CHARS = "\"'“”"