import pysbd

from tts_audiobook_tool.sentence_segmenter import SentenceSegmenter
from tts_audiobook_tool.text_segment import TextSegment

class TextSegmenter:

    @staticmethod
    def segment_text(full_text: str, max_words: int, language="en") -> list[TextSegment]:
        """
        Segments full project text into chunks.
        Segments by sentence. When sentence is longer than max_words, splits sentence into chunks.
        """

        # Segment text into sentences using pysbd
        segmenter = pysbd.Segmenter(language=language, clean=False, char_span=False) # clean=False - important
        texts = segmenter.segment(full_text)

        # pysbd treats quotes with multiple sentences as a single sentence, so split them up
        new_texts = []
        for text in texts:
            if is_double_quote(text):
                lst = segment_quote(text, segmenter)
                new_texts.extend(lst)
            else:
                new_texts.append(text)
        texts = new_texts

        # Split long sentences using own algo
        new_texts = []
        for text in texts:
            lst = SentenceSegmenter.segment_sentence(text, max_words=max_words)
            new_texts.extend(lst)
        texts = new_texts

        counter = 0
        result = []
        for text in texts:
            length = len(text)
            text_segment = TextSegment(text, counter, counter + length)
            result.append(text_segment)
            counter += length

        return result

# ---

def is_double_quote(s: str) -> bool:
    """
    Returns True if string starts and ends with a double-quote character, whitespace notwithstanding
    """
    s = s.strip()
    if len(s) <= 3:
        return False
    first = s[0]
    last = s[-1]
    return first in '"＂″‶〝〞' and last in '"＂″‶〝〞'


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
