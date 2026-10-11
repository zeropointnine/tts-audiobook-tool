from __future__ import annotations
import json

from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.util import *

class TimedPhrase:
    """
    A phrase (piece of text) with a start and end time.
    Used for the app metadata; the text gets displayed as-is in the player UI.

    `reason` is why the text was split off at its end (see `Reason`). It is optional:
    most users of this class (app metadata, the player) neither have nor need it.
    The segment sidecar JSON records it, so later processing of a saved segment can
    tell sentence boundaries from phrase boundaries without the project text.
    It is only written to a dict when asked for (`to_dict(..., include_reason=True)`),
    so the app metadata format is unaffected. None means unknown (e.g. a sidecar
    written before it was recorded).
    """

    # Class-level default so TimedPhrase objects unpickled from before this
    # attribute existed (enhance artifacts) still have it.
    reason: Reason | None = None

    def __init__(self, text: str, time_start: float, time_end: float, reason: Reason | str | None = None):
        self.text = text
        self.time_start = time_start
        self.time_end = time_end
        # A str is the JSON form, as read from a sidecar
        self.reason = Reason.from_json_value(reason) if isinstance(reason, str) else reason

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
            time_end=time_end,
            reason=phrase.reason,
        )

    @staticmethod
    def to_dict(timed_text_segment: TimedPhrase, include_reason: bool = False) -> dict:
        """
        `include_reason` adds a "reason" key (when the reason is known). Leave it off
        for the app metadata, which has a fixed format.
        """
        d = {
            "text": timed_text_segment.text,
            "time_start": timed_text_segment.time_start,
            "time_end": timed_text_segment.time_end
        }
        if include_reason and timed_text_segment.reason is not None:
            d["reason"] = timed_text_segment.reason.json_value
        return d

    @staticmethod
    def timed_phrases_to_dicts(timed_text_segments: list[TimedPhrase], include_reason: bool = False) -> list[dict]:
        result = []
        for item in timed_text_segments:
            result.append(TimedPhrase.to_dict(item, include_reason))
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
