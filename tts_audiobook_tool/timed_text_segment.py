from __future__ import annotations
import json

from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *

class TimedTextSegment:
    """
    """

    def __init__(self, text: str, index_start: int, index_end: int, time_start: float, time_end: float):
        self.text = text
        self.index_start = index_start
        self.index_end = index_end # is exclusive
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

    @staticmethod
    def list_from_dict_list(dicts: list[dict]) -> list[TimedTextSegment] | str:
        """
        Returns list or error string
        """
        result = []
        for d in dicts:
            try:
                result.append( TimedTextSegment(**d) )
            except Exception as e:
                return f"Error with dict {json.dumps(d)} - {e}"
        return result

    def pretty_string(self, index: int=-1, use_error_color: bool=True) -> str:
        if index >= 0:
            s1 = f"[{str(index).rjust(5)}] "
        else:
            s1 = ""
        s2 = f"{str(self.index_start).rjust(5)}-{str(self.index_end).ljust(5)}  "
        if use_error_color and self.time_start == 0 and self.time_end == 0:
            s3 = f"{COL_ERROR}{time_stamp(self.time_start)}-{time_stamp(self.time_end)}{Ansi.RESET}  "
        else:
            s3 = f"{time_stamp(self.time_start)}-{time_stamp(self.time_end)}  "
        s4 = ellipsize(self.text.strip(), 50)
        return f"{s1}{s2}{s3}{s4}"

    @staticmethod
    def make_list_using(text_segments: list[TextSegment], durations: list[float]) -> list[TimedTextSegment]:

        if len(text_segments) != len(durations):
            raise ValueError(f"Parallel arrays have different lengths {len(text_segments)} {len(durations)}")

        timed_text_segments = []
        total_seconds = 0.0

        for i in range(len(text_segments)):

            text_segment = text_segments[i]
            duration = durations[i]

            if not duration:
                timed_text_segment = TimedTextSegment.make_using(text_segment, 0, 0) # has no start/end times
                timed_text_segments.append(timed_text_segment)
                continue

            timed_text_segment = TimedTextSegment.make_using(text_segment, total_seconds, total_seconds + duration)
            timed_text_segments.append(timed_text_segment)
            total_seconds += duration

        return timed_text_segments

    @staticmethod
    def get_discontinuities(items: list[TimedTextSegment]) -> list[tuple[int, int]]:
        """
        Returns a list of index ranges where 2 or more consecutive items have a time_start and time_end of 0
        (making the assumption that just one zeroed item is 'non-verbal', formatting-related text)
        """
        discontinuities = []
        current_discontinuity_start = -1

        for i, item in enumerate(items):
            is_discontinuous = (item.time_start == 0.0 and item.time_end == 0.0)

            if is_discontinuous:
                if current_discontinuity_start == -1:
                    current_discontinuity_start = i
            else:
                if current_discontinuity_start != -1:
                    # End of a discontinuity sequence
                    if i - current_discontinuity_start >= 2:
                        discontinuities.append((current_discontinuity_start, i - 1))
                    current_discontinuity_start = -1

        # Check for a discontinuity sequence at the end of the list
        if current_discontinuity_start != -1:
            if len(items) - current_discontinuity_start >= 2:
                discontinuities.append((current_discontinuity_start, len(items) - 1))

        return discontinuities
