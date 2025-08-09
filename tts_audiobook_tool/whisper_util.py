
from typing import Iterable

import librosa
import numpy as np
from tts_audiobook_tool.app_types import Sound, Word
from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.tts import Tts
from typing import Iterable, TYPE_CHECKING # type: ignore
from faster_whisper.transcribe import Segment


class WhisperUtil:

    @staticmethod
    def transcribe(sound: Sound) -> list[Word] | str:
        """
        Transcribes the audio data. Makes temporary resampled audio if necessary.
        Returns flat list of Words, discarding other Segment info and the info object.
        Or returns error string on fail.
        """
        if sound.sr != WHISPER_SAMPLERATE:
            sound = WhisperUtil.resample_sound_for_whisper(sound)

        try:
            segments, _ = Tts.get_whisper().transcribe(audio=sound.data, word_timestamps=True, language=None)
        except Exception as e:
            return str(e)

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

    # ---

    @staticmethod
    def resample_sound_for_whisper(sound: Sound) -> Sound:
        data = sound.data
        data = np.nan_to_num(sound.data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -1.0, 1.0)
        data = librosa.resample(data, orig_sr=sound.sr, target_sr=WHISPER_SAMPLERATE)
        return Sound(data, WHISPER_SAMPLERATE)
