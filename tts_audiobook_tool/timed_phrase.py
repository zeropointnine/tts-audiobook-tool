from __future__ import annotations
import json

from tts_audiobook_tool.phrase import Phrase
from tts_audiobook_tool.util import *

class TimedPhrase:
    """
    A phrase (piece of text) with a start and end time.
    Used for the app metadata; the text gets displayed as-is in the player UI.
    Note how does not have/need 'reason' at this stage in teh pipeline.
    """

    def __init__(self, text: str, time_start: float, time_end: float):
        self.text = text
        self.time_start = time_start
        self.time_end = time_end

    def __str__(self) -> str:
        s1 = ellipsize(self.text.strip(), 50)
        s2 = f"{time_stamp(self.time_start)}-{time_stamp(self.time_end)} "
        return f"[TimedPhrase] {s1} {s2}"


    @property
    def presentable_text(self) -> str:
        """ Text in 'presentable' format for UI-related purposes """
        return self.text.strip()

    # ---

    @staticmethod
    def make_using(phrase: Phrase, time_start: float, time_end: float) -> TimedPhrase:
        return TimedPhrase(
            text=phrase.text,
            time_start=time_start,
            time_end=time_end
        )

    @staticmethod
    def to_dict(timed_text_segment: TimedPhrase) -> dict:
        return {
            "text": timed_text_segment.text,
            "time_start": timed_text_segment.time_start,
            "time_end": timed_text_segment.time_end
        }

    @staticmethod
    def timed_phrases_to_dicts(timed_text_segments: list[TimedPhrase]) -> list[dict]:
        result = []
        for item in timed_text_segments:
            result.append(TimedPhrase.to_dict(item))
        return result

    @staticmethod
    def dicts_to_timed_phrases(dicts: list[dict]) -> list[TimedPhrase] | str:
        """
        Returns list or error string
        """
        result = []
        for d in dicts:
            try:
                result.append( TimedPhrase(**d) )
            except Exception as e:
                return f"Error with dict {json.dumps(d)} - {e}"
        return result

    @staticmethod
    def make_list_using(phrases: list[Phrase], durations: list[float]) -> list[TimedPhrase]:

        if len(phrases) != len(durations):
            raise ValueError(f"Parallel arrays have different lengths {len(phrases)} {len(durations)}")

        timed_text_segments = []
        seconds_cursor = 0.0

        for i in range(len(phrases)):

            text_segment = phrases[i]
            duration = durations[i]

            if not duration:
                timed_text_segment = TimedPhrase.make_using(text_segment, 0, 0) # has no start/end times
                timed_text_segments.append(timed_text_segment)
                continue

            timed_text_segment = TimedPhrase.make_using(text_segment, seconds_cursor, seconds_cursor + duration)
            timed_text_segments.append(timed_text_segment)
            seconds_cursor += duration

        return timed_text_segments

    @staticmethod
    def get_discontinuities(items: list[TimedPhrase], filter_num_consecutive:int=1) -> list[tuple[int, int]]:
        """
        Returns a list of index ranges where consecutive items have a time_start and time_end of 0
        (* making the assumption that just one zeroed item is 'non-verbal', formatting-related text)
        TODO: brittle assumption!
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
                    if i - current_discontinuity_start >= filter_num_consecutive:
                        discontinuities.append((current_discontinuity_start, i - 1))
                    current_discontinuity_start = -1

        # Test last
        if current_discontinuity_start != -1:
            if len(items) - current_discontinuity_start >= filter_num_consecutive:
                discontinuities.append((current_discontinuity_start, len(items) - 1))

        return discontinuities
