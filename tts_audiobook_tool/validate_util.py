from __future__ import annotations

from tts_audiobook_tool.app_types import FailResult, PassResult, Sound, TrimmableResult, ValidationResult, Word
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class ValidateUtil:

    @staticmethod
    def validate_item(
        sound: Sound, 
        source: str, 
        transcript_words: list[Word],
        project_language_code: str
    ) -> ValidationResult:
        """
        Tests if TTS generation is ~valid, and returns a ValidationResult 
        (PassResult, TrimmableResult, FailResult)
        """
        transcript = WhisperUtil.get_flat_text_from_words(transcript_words)
        is_fail, num_word_fails, word_fail_threshold = TranscribeUtil.is_word_failure(
            source, transcript, is_loose=False, language_code=project_language_code
        )
        if is_fail:
            return FailResult(
                transcript_words=transcript_words, message=f"Too many word failures",
                num_word_fails=num_word_fails, word_fail_threshold=word_fail_threshold
            )

        # TODO: P0 Update count_word_failures and apply to this to make more robust. Or drop altogether.
        SUBSTRING_TRIM_ENABLED = True
        if SUBSTRING_TRIM_ENABLED:

            # Substring test
            opt_timestamps = TranscribeUtil.get_substring_time_range(source, transcript_words)
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
                    transcript_words=transcript_words,
                    start_time=start_time,
                    end_time=end_time,
                    duration=sound.duration
                )

            # Test for excess audio before or after audio-text which is not re
            trim_start_time = TranscribeUtil.get_semantic_match_start_time_trim(
                source, transcript_words, sound
            )
            if trim_start_time is not None:
                trim_start_time = SoundUtil.get_local_minima(sound, trim_start_time)
                if trim_start_time == 0:
                    trim_start_time = None

            # Currently disabled because end-time is so unreliable with current whisper implementation,
            # does almost more harm than good.
            # Although still worth using for 'substring test' maybe

            # trim_end_time = TranscribeUtil.get_semantic_match_end_time_trim(
            #     reference_text, transcribed_words, sound, include_last_word=tts_specs.semantic_trim_last
            # )
            trim_end_time= None

            if trim_end_time is not None:
                trim_end_time = SoundUtil.get_local_minima(sound, trim_end_time)
                if abs(sound.duration - trim_end_time) < 0.05: # ~epsilon
                    trim_end_time = None

            if trim_start_time is not None or trim_end_time is not None:
                return TrimmableResult(
                    transcript_words=transcript_words,
                    start_time=trim_start_time,
                    end_time=trim_end_time,
                    duration=sound.duration
                )

        # At this point we consider the item to have "passed"
        return PassResult(transcript_words, num_word_fails, word_fail_threshold)

    @staticmethod
    def is_unsupported_language_code(code: str) -> bool:
        for item in VALIDATION_UNSUPPORTED_LANGUAGES:
            if code.startswith(item):
                return True
        return False
