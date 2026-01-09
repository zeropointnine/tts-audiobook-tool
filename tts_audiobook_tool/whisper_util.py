import time
from typing import Iterable, TYPE_CHECKING # type: ignore

import librosa
import numpy as np

from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant, Word
from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.stt import Stt
from faster_whisper.transcribe import Segment

from tts_audiobook_tool.util import make_error_string


class WhisperUtil:

    @staticmethod
    def transcribe_to_words(
            sound: Sound,
            language_code: str,
            stt_variant: SttVariant = SttVariant.LARGE_V3,
            stt_config: SttConfig = SttConfig.CUDA_FLOAT16,
    ) -> list[Word] | str:
        """
        All whisper transcription should be done through here.

        Simple wrapper around whisper `transcribe()`.
        Returns list of (whisper) Words or error string on fail.

        Makes temporary resampled audio if necessary.
        """
        if sound.sr != WHISPER_SAMPLERATE:
            sound = WhisperUtil.resample_sound_for_whisper(sound)

        Stt.set_variant(stt_variant)
        Stt.set_config(stt_config)

        if language_code and language_code not in Stt.get_whisper().supported_languages:
            # Silently remove language code rather than triggering an exception
            language_code = ""

        try:
            segments, _ = Stt.get_whisper().transcribe(audio=sound.data, word_timestamps=True, language=language_code or None)

            # Convert generator to concrete list (does the actual inference)
            segments = list(segments)

        except Exception as e:
            return make_error_string(e)

        # Flatten Segments into Words
        words = WhisperUtil.get_words_from_segments(segments)
        return words

    # ---

    @staticmethod
    def get_words_from_segments(segments: Iterable[Segment]) -> list[Word]:
        """
        Converts an interable of faster-whisper Segments into a flattened list of Words.
        """
        words = []
        for segment in segments:
            if segment.words:  # Ensure the words list exists and is not empty
                words.extend(segment.words)
        return words


    @staticmethod
    def get_flat_text_from_segments(segments: Iterable[Segment]) -> str:
        words = WhisperUtil.get_words_from_segments(segments)
        return WhisperUtil.get_flat_text_from_words(words)

    @staticmethod
    def get_flat_text_from_words(words: list[Word]) -> str:
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
            text = WhisperUtil.get_flat_text_from_words(words)
        else:
            words = [word for word in words if word.probability >= min_probability]
            text = WhisperUtil.get_flat_text_from_words(words)
        return text
    
    @staticmethod
    def words_to_json(words: list[Word], include_probability: bool=False) -> list[dict]:
        results = []
        for word in words:
            d = { "start": word.start, "end": word.end, "word": word.word }
            if include_probability:
                d["probability"] = word.probability
            results.append(d)
        return results

    # ---

    @staticmethod
    def resample_sound_for_whisper(sound: Sound) -> Sound:
        data = sound.data
        data = np.nan_to_num(sound.data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -1.0, 1.0)
        data = librosa.resample(data, orig_sr=sound.sr, target_sr=WHISPER_SAMPLERATE)
        return Sound(data, WHISPER_SAMPLERATE)
    