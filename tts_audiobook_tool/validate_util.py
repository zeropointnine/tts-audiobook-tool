from __future__ import annotations

from tts_audiobook_tool.app_types import FailResult, PassResult, Sound, TrimmableResult, ValidationResult
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.tts_info import TtsInfo
from tts_audiobook_tool.util import *

class ValidateUtil:

    @staticmethod
    def validate_item(sound: Sound, reference_text: str, whisper_data: dict, tts_specs: TtsInfo) -> ValidationResult:

        # Runs various tests to determine if audio generation seems to be valid.
        # Tests err on the conservative side and prioritize avoiding false positives
        # Think layers of swiss cheese mkay.
        # Order of tests matter here.

        # Static audio test
        is_static = TranscribeUtil.is_audio_static(sound, whisper_data)
        if is_static:
            return FailResult("Audio is static")

        # Substring test
        timestamps = TranscribeUtil.get_substring_time_range(reference_text, whisper_data)
        if timestamps:
            return TrimmableResult(
                "Excess words found at start and/or end, but valid substring exists",
                timestamps[0], timestamps[1], duration=sound.duration
            )

        transcribed_text = TranscribeUtil.get_whisper_data_text(whisper_data)

        # Repeat phrases test
        repeats = TranscribeUtil.find_bad_repeats(reference_text, whisper_data)
        if repeats:
            s = next(iter(repeats))
            if len(repeats) > 1:
                s += ", ..."
            return FailResult(f"Repeated word/phrase: {s}")

        # Repeat word count test (tries to detect same issue as above)
        num_over_occurrences = TranscribeUtil.num_bad_over_occurrences(reference_text, whisper_data)
        if num_over_occurrences:
            return FailResult(f"Words over-occurrence count: {num_over_occurrences}")

        # Dropped words at end or beginning
        if TranscribeUtil.is_drop_fail_tail(reference_text, transcribed_text): # TODO pass along more info here
            return FailResult("Dropped phrase at end")

        # Word count delta test
        fail_reason = TranscribeUtil.is_word_count_fail(reference_text, whisper_data)
        if fail_reason:
            return FailResult(fail_reason)

        # Excess audio test
        trim_start_time = TranscribeUtil.get_semantic_match_start_time_trim(reference_text, whisper_data, sound)
        trim_end_time = TranscribeUtil.get_semantic_match_end_time_trim(
            reference_text, whisper_data, sound, include_last_word=tts_specs.semantic_trim_last
        )
        messages = []
        if trim_start_time:
            messages.append(f"Found excess audio at start.")
        if trim_end_time:
            messages.append(f"Found excess audio at end.")
        if trim_start_time or trim_end_time:
            return TrimmableResult(", ".join(messages), trim_start_time, trim_end_time, sound.duration)

        # At this point we consider the item to have "passed"
        return PassResult()

# ---

