import time
from typing import Iterable, TYPE_CHECKING # type: ignore

import librosa
import numpy as np

from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from faster_whisper.transcribe import Segment

from tts_audiobook_tool.util import make_error_string


class WhisperUtil:

    @staticmethod
    def transcribe_to_segments(sound: Sound) -> list[Segment] | str:
        """
        Simple wrapper around whisper `transcribe()`.
        Returns the segments generator or error string on fail.

        Makes temporary resampled audio if necessary.
        """
        if sound.sr != WHISPER_SAMPLERATE:
            sound = WhisperUtil.resample_sound_for_whisper(sound)

        try:
            segments, _ = Stt.get_whisper().transcribe(audio=sound.data, word_timestamps=True, language=None)
        except Exception as e:
            return make_error_string(e)

        segments = list(segments)
        return segments

    @staticmethod
    def transcribe_to_words(sound: Sound) -> list[Word] | str:
        """
        Returns flattened list of Words
        """
        result = WhisperUtil.transcribe_to_segments(sound)
        if isinstance(result, str):
            return result
        segments = result

        words = WhisperUtil.get_words_from_segments(segments)
        return words

    # ---

    @staticmethod
    def get_words_from_segments(segments: Iterable[Segment]) -> list[Word]:
        """
        Converts a generator of faster-whisper Segments into a flattened list of Words.
        """
        words = []
        for segment in segments:
            if segment.words:  # Ensure the words list exists and is not empty
                words.extend(segment.words)
        return words

    @staticmethod
    def get_flat_text_from_segments(segments: Iterable[Segment]) -> str:
        words = WhisperUtil.get_words_from_segments(segments)
        return WhisperUtil.get_flat_text(words)

    @staticmethod
    def get_flat_text(words: list[Word]) -> str:
        """
        Returns join'ed word.words.
        Rem, this is well formatted, retaining punctuation and capitalizations.
        """
        text = " ".join( [word.word.strip() for word in words] )
        return text

    @staticmethod
    def get_flat_text_filtered_by_probability(words: list[Word], min_probability: float) -> str:
        """
        Returns joined words, but only the high-confidence ones.
        For use with voice clone transcription, where it's better to omit words entirely
        when they are not of high-ish confidence, apparently.
        """
        if min_probability <= 0.0:
            text = WhisperUtil.get_flat_text(words)
        else:
            words = [word for word in words if word.probability >= min_probability]
            text = WhisperUtil.get_flat_text(words)
        return text

    # ---

    @staticmethod
    def resample_sound_for_whisper(sound: Sound) -> Sound:
        data = sound.data
        data = np.nan_to_num(sound.data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -1.0, 1.0)
        data = librosa.resample(data, orig_sr=sound.sr, target_sr=WHISPER_SAMPLERATE)
        return Sound(data, WHISPER_SAMPLERATE)
