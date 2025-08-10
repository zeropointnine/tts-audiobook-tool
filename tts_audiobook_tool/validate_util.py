from __future__ import annotations

from tts_audiobook_tool.app_types import FailResult, PassResult, Sound, TrimmableResult, ValidationResult, Word
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.tts_info import TtsInfo
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class ValidateUtil:

    @staticmethod
    def validate_item(sound: Sound, reference_text: str, transcribed_words: list[Word], tts_specs: TtsInfo) -> ValidationResult:

        # Runs various tests to determine if audio generation seems to be valid.
        # Errs on the conservative side, prioritizes avoiding false positives.
        # Think layers of swiss cheese mkay.
        # Order of tests matter here.

        transcribed_text = WhisperUtil.get_flat_text(transcribed_words)

        # Static audio test
        is_static = TranscribeUtil.is_audio_static(sound, transcribed_text)
        if is_static:
            return FailResult("Audio is static")

        # Substring test
        opt_timestamps = TranscribeUtil.get_substring_time_range(reference_text, transcribed_words)
        if opt_timestamps:
            start_time, end_time = opt_timestamps
            if start_time > 0:
                start_time = SoundUtil.get_local_minima(sound, start_time)
            if end_time < sound.duration:
                end_time = SoundUtil.get_local_minima(sound, end_time)
            if start_time == 0:
                start_time = None
            if end_time == sound.duration:
                end_time = None
            return TrimmableResult(
                "Audio contains excess words but reference text exists as substring",
                start_time, end_time, duration=sound.duration
            )

        # Repeat phrases test
        repeats = TranscribeUtil.find_bad_repeats(reference_text, transcribed_text)
        if repeats:
            s = next(iter(repeats))
            if len(repeats) > 1:
                s += ", ..."
            return FailResult(f"Repeated word/phrase: {s}")

        # Repeat word count test (tries to detect same issue as above)
        num_over_occurrences = TranscribeUtil.num_bad_over_occurrences(reference_text, transcribed_text)
        if num_over_occurrences:
            return FailResult(f"Words over-occurrence count: {num_over_occurrences}")

        # Dropped words at end or beginning
        if TranscribeUtil.is_drop_fail_tail(reference_text, transcribed_text): # TODO pass along more info here for the fail message
            return FailResult("Missing word/s at end")

        # Word count delta test
        fail_reason = TranscribeUtil.is_word_count_fail(reference_text, transcribed_text)
        if fail_reason:
            return FailResult(fail_reason)

        # Test for excess audio before or after audio-text
        trim_start_time = TranscribeUtil.get_semantic_match_start_time_trim(
            reference_text, transcribed_words, sound
        )
        if trim_start_time is not None:
            trim_start_time = SoundUtil.get_local_minima(sound, trim_start_time)
            if trim_start_time == 0:
                trim_start_time = None

        trim_end_time = TranscribeUtil.get_semantic_match_end_time_trim(
            reference_text, transcribed_words, sound, include_last_word=tts_specs.semantic_trim_last
        )
        if trim_end_time is not None:
            trim_end_time = SoundUtil.get_local_minima(sound, trim_end_time)
            if abs(sound.duration - trim_end_time) < 0.05: # ~epsilon
                trim_end_time = None

        if trim_start_time is not None or trim_end_time is not None:
            messages = []
            if trim_start_time is not None:
                messages.append(f"Found excess audio at start")
            if trim_end_time is not None:
                messages.append(f"Found excess audio at end")
            message = ", ".join(messages)
            return TrimmableResult(message, trim_start_time, trim_end_time, sound.duration)

        # At this point we consider the item to have "passed"
        return PassResult()
