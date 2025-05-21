from __future__ import annotations
from typing import NamedTuple

class TextSegment:
    def __init__(self, text: str, index_start: int, index_end: int):
        self.text = text
        self.index_start = index_start
        self.index_end = index_end # exclusive, not inclusive

    @staticmethod
    def to_dict(text_segment: TextSegment) -> dict:
        return {
            "text": text_segment.text,
            "index_start": text_segment.index_start,
            "index_end": text_segment.index_end
        }

    @staticmethod
    def to_dict_list(text_segments: list[TextSegment]) -> list[dict]:
        result = []
        for item in text_segments:
            result.append(TextSegment.to_dict(item))
        return result

class TimedTextSegment:
    def __init__(self, text: str, index_start: int, index_end: int, time_start: float, time_end: float):
        self.text = text
        self.index_start = index_start
        self.index_end = index_end # exclusive, not inclusive
        self.time_start = time_start
        self.time_end = time_end

    @staticmethod
    def make_using(text_segment: TextSegment, time_start: float, time_end: float) -> TimedTextSegment:
        return TimedTextSegment(
            text=text_segment.text,
            index_start=text_segment.index_start,
            index_end=text_segment.index_end,
            time_start=time_start,
            time_end=time_end
        )

    @staticmethod
    def to_dict(timed_text_segment: TimedTextSegment) -> dict:
        return {
            "text": timed_text_segment.text,
            "index_start": timed_text_segment.index_start,
            "index_end": timed_text_segment.index_end,
            "time_start": timed_text_segment.time_start,
            "time_end": timed_text_segment.time_end
        }

    @staticmethod
    def to_dict_list(timed_text_segments: list[TimedTextSegment]) -> list[dict]:
        result = []
        for item in timed_text_segments:
            result.append(TimedTextSegment.to_dict(item))
        return result