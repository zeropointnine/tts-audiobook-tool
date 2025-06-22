from typing import NamedTuple

from numpy import ndarray

from tts_audiobook_tool.timed_text_segment import TimedTextSegment


class AppMetadata(NamedTuple):
    raw_text: str
    timed_text_segments: list[TimedTextSegment]


class Sound(NamedTuple):
    data: ndarray
    sr: int

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr