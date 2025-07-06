from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple
from tts_audiobook_tool.app_types import Sound, TtsInfo
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.util import *

class ValidateUtil:

    @staticmethod
    def validate_item(sound: Sound, reference_text: str, whisper_data: dict, tts_specs: TtsInfo) -> ValidateResult:

        # Runs various tests to determine if audio generation seems to be valid.
        # Tests err on the conservative side and prioritize avoiding false positives
        # Think layers of swiss cheese mkay.
        # Order of tests matter here.

        # Static audio test
        is_static = TranscribeUtil.is_audio_static(sound, whisper_data)
        if is_static:
            return ValidateResult(ValidateResultType.INVALID, "Audio is static")

        # Substring test (fixable)
        timestamps = TranscribeUtil.get_substring_time_range(reference_text, whisper_data)
        if timestamps:
            message = f"Excess words detected, substring at {timestamps[0]:.2f}-{timestamps[1]:.2f}"
            result = ValidateResult(
                ValidateResultType.TRIMMABLE,
                message,
                timestamps[0],
                timestamps[1]
            )
            return result

        transcribed_text = TranscribeUtil.get_whisper_data_text(whisper_data)

        # Repeat phrases test
        repeats = TranscribeUtil.find_bad_repeats(reference_text, whisper_data)
        if repeats:
            s = next(iter(repeats))
            if len(repeats) > 1:
                s += ", ..."
            return ValidateResult(ValidateResultType.INVALID, f"Repeated word/phrase: {s}")

        # Repeat word count test (tries to detect same issue as above)
        num_over_occurrences = TranscribeUtil.num_bad_over_occurrences(reference_text, whisper_data)
        if num_over_occurrences:
            return ValidateResult(ValidateResultType.INVALID, f"Words over-occurrence count: {num_over_occurrences}")

        # Dropped words at end or beginning
        if TranscribeUtil.is_drop_fail_tail(reference_text, transcribed_text):
            return ValidateResult(ValidateResultType.INVALID, "Dropped phrase at end")

        # Word count delta test
        fail_reason = TranscribeUtil.is_word_count_fail(reference_text, whisper_data)
        if fail_reason:
            return ValidateResult(ValidateResultType.INVALID, fail_reason)

        # Excess audio
        trim_start_time = TranscribeUtil.get_semantic_match_start_time_trim(reference_text, whisper_data, sound)
        trim_end_time = TranscribeUtil.get_semantic_match_end_time_trim(
            reference_text, whisper_data, sound, include_last_word=tts_specs.semantic_trim_last
        )

        messages = []
        if trim_start_time:
            messages.append(f"Excess at start ({(trim_start_time):.2f}s)")
        if trim_end_time:
            messages.append(f"Excess at end ({(sound.duration - trim_end_time):.2f}s)")

        if trim_start_time or trim_end_time:
            result = ValidateResult(
                ValidateResultType.TRIMMABLE,
                ", ".join(messages),
                trim_start_time,
                trim_end_time
            )
            return result

        # At this point we consider the item to have "passed"
        return ValidateResult(ValidateResultType.VALID, "Passed validation tests")

# ---

class ValidateResult(NamedTuple):
    result: ValidateResultType
    message: str
    trim_start: float | None = None
    trim_end: float | None = None

class ValidateResultType(Enum):
    VALID = auto()
    TRIMMABLE = auto()
    INVALID = auto()
