from __future__ import annotations
from enum import Enum, auto
import json

from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *

class TextSegment:
    """
    """

    def __init__(self, text: str, index_start: int, index_end: int, reason: TextSegmentReason):
        self.text = text
        self.index_start = index_start
        self.index_end = index_end # is exclusive
        self.reason = reason

    def __str__(self):
        return f"TextSegment: {str(self.index_start).rjust(6)}-{str(self.index_end).ljust(6)} {str(self.reason.value).ljust(4)} text: {self.text.strip()}"

    @staticmethod
    def to_dict(text_segment: TextSegment) -> dict:
        return {
            "text": text_segment.text,
            "index_start": text_segment.index_start,
            "index_end": text_segment.index_end,
            "reason": text_segment.reason.value
        }

    @staticmethod
    def list_to_dict_list(text_segments: list[TextSegment]) -> list[dict]:
        result = []
        for item in text_segments:
            result.append(TextSegment.to_dict(item))
        return result

    @staticmethod
    def dict_list_to_list(object: list[dict]) -> list[TextSegment]:
        if not object or not isinstance(object, list):
            L.e(f"bad text segments object: {object}")
            return []

        text_segments = []

        for item in object:
            if not isinstance(item, dict):
                L.e(f"bad type: {item}")
                return []
            if not "text" in item or not "index_start" in item or not "index_end" in item:
                L.e(f"missing required property in item: {item}")
                return []

            try:
                start = int(item["index_start"])
                end = int(item["index_end"])
            except:
                L.e(f"parse float error: {item}")
                return []

            try:
                reason = TextSegmentReason(item["reason"])
            except:
                reason = TextSegmentReason.UNDEFINED

            text_segment = TextSegment(item["text"], start, end, reason)
            text_segments.append(text_segment)
        return text_segments


class TextSegmentReason(Enum):
    """
    Reason for the text segment being split

    (Using enum for potential expandability)
    """
    UNDEFINED = "undefined" # for back-compat
    SENTENCE = "s"
    INSIDE_SENTENCE = "is"