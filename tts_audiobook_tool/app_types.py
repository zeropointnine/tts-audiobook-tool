from typing import NamedTuple

from tts_audiobook_tool.timed_text_segment import TimedTextSegment


class AppMetadata(NamedTuple):
    raw_text: str
    timed_text_segments: list[TimedTextSegment]