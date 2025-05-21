from typing import List, cast
import pysbd
from pysbd.utils import TextSpan

from tts_audiobook_tool.sentence_segmenter import SentenceSegmenter
from tts_audiobook_tool.text_segment import TextSegment

class TextSegmenter:

    @staticmethod
    def segment_text(text: str, max_words: int, language="en") -> list[TextSegment]:

        # Use pysbd to segment text into sentences
        segmenter = pysbd.Segmenter(language=language, clean=False, char_span=True) # clean=False - important
        text_spans: list[TextSpan] = segmenter.segment(text) # type: ignore
        # And convert to in-house type
        text_segments = [TextSegment(item.sent, item.start, item.end) for item in text_spans ]

        # Split sentences if needed
        results = []
        for text_segment in text_segments:
            fragments = SentenceSegmenter.segment_sentence(text_segment.text, max_words=max_words)
            index = text_segment.index_start
            for fragment in fragments:
                result = TextSegment(fragment, index, index + len(fragment))
                index += len(fragment)
                results.append(result)

        return results
