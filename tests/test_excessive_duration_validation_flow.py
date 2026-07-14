from unittest.mock import patch

import numpy as np

from tts_audiobook_tool.app_types import ConcreteWord, Sound, Strictness, Word
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.validation_result import MusicFailResult, ExcessiveDurationResult, WordErrorResult
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.validator import Validator


def make_sound(duration: float) -> Sound:
    sr = 100
    return Sound(np.zeros(round(duration * sr), dtype=np.float32), sr)


def make_word(text: str, index: int) -> ConcreteWord:
    return ConcreteWord(start=float(index), end=float(index) + 0.5, word=text, probability=1.0)


def make_words(*items: str) -> list[Word]:
    return [make_word(text, index) for index, text in enumerate(items)]


def test_sus_duration_threshold_is_strictly_greater_than_calculated_limit() -> None:
    assert not ExcessiveDurationResult.is_excessively_long("alpha beta gamma", "en", 3.75)
    assert ExcessiveDurationResult.is_excessively_long("alpha beta gamma", "en", 3.76)


def test_sus_duration_allows_source_text_with_long_numbers() -> None:
    assert not ExcessiveDurationResult.is_excessively_long("flight 123 lasted", "en", 10.0)


def test_validate_returns_sus_duration_after_music_check_when_transcript_duration_is_too_long() -> None:
    words = make_words("hello", "world")
    sound = make_sound(3.01)

    with patch("tts_audiobook_tool.validator.ModelManager.has_yamnet_detector", return_value=False), \
            patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE):
        result = Validator.validate(sound, "hello world", words, "en", Strictness.INTOLERANT)

    assert isinstance(result, ExcessiveDurationResult)
    assert result.sound is sound
    assert result.transcript_words == words
    assert result.duration == sound.duration
    assert result.is_fail


def test_validate_preserves_music_precedence_over_sus_duration() -> None:
    words = make_words("hello")
    sound = make_sound(2.26)

    class Detector:
        def has_music(self, sound: Sound, threshold: float) -> bool:
            return True

    with patch("tts_audiobook_tool.validator.ModelManager.has_yamnet_detector", return_value=True), \
            patch("tts_audiobook_tool.validator.ModelManager.get_yamnet_detector", return_value=Detector()):
        result = Validator.validate(sound, "hello", words, "en", Strictness.INTOLERANT)

    assert isinstance(result, MusicFailResult)


def test_validate_returns_word_error_result_when_duration_is_not_suspicious() -> None:
    words = make_words("hello", "world")
    sound = make_sound(3.0)

    with patch("tts_audiobook_tool.validator.ModelManager.has_yamnet_detector", return_value=False), \
            patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE):
        result = Validator.validate(sound, "hello world", words, "en", Strictness.INTOLERANT)

    assert isinstance(result, WordErrorResult)
    assert not result.is_fail


def test_sus_duration_uses_all_or_nothing_word_error_sentinel_without_music_exception() -> None:
    words = make_words("hello")
    result = ExcessiveDurationResult(make_sound(2.26), words, duration=2.26)
    with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE):
        project = Project.model_validate({"language_code": "en"})
    phrase_group = PhraseGroup([Phrase("hello", Reason.SENTENCE)])

    assert SegmentTranscriptUtil.make_generation_word_error_count(result) == 99

    info = SegmentTranscriptUtil.from_validation_result(
        project=project,
        phrase_group=phrase_group,
        index=0,
        validation_result=result,
    )
    assert info.generation_word_error_count == 99
    assert info.exception == SegmentTranscriptUtil.EXCEPTION_EXCESSIVE_DURATION


def test_music_exception_semantics_are_preserved() -> None:
    words = make_words("hello")
    result = MusicFailResult(make_sound(1.0), words)
    with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE):
        project = Project.model_validate({"language_code": "en"})
    phrase_group = PhraseGroup([Phrase("hello", Reason.SENTENCE)])

    info = SegmentTranscriptUtil.from_validation_result(
        project=project,
        phrase_group=phrase_group,
        index=0,
        validation_result=result,
    )

    assert info.generation_word_error_count == 99
    assert info.exception == SegmentTranscriptUtil.EXCEPTION_MUSIC_DETECTED


def test_sus_duration_file_name_uses_99_fail_tag() -> None:
    words = make_words("hello")
    result = ExcessiveDurationResult(make_sound(2.26), words, duration=2.26)
    with patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE):
        project = Project.model_validate({"language_code": "en"})
    phrase_group = PhraseGroup([Phrase("hello", Reason.SENTENCE)])

    file_name = SoundSegmentUtil.make_file_name(
        index=0,
        phrase_group=phrase_group,
        project=project,
        tts_model_type=TtsModelType.NONE.value,
        validation_result=result,
        is_real_time=False,
        voice_tag="test-voice",
    )

    assert " [99] " in file_name
