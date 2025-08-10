import time
from typing import Iterable, TYPE_CHECKING # type: ignore

import librosa
import numpy as np

from tts_audiobook_tool.app_types import ConcreteWord, Sound, Word
from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.tts import Tts
from faster_whisper.transcribe import Segment


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
            segments, _ = Tts.get_whisper().transcribe(audio=sound.data, word_timestamps=True, language=None)
        except Exception as e:
            return str(e)

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

    @staticmethod
    def make_aligned_words(sound: Sound, segments_list: list[Segment]) -> list[Word]:
        """ TODO: Untested because dependency incompatibility hell. """

        import whisperx # type: ignore

        # Prepare the segments in the format whisperx expects
        whisper_results = { "segments": [] }
        for segment in segments_list:
            segment_dict = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "words": [ { "word": w.word, "start": w.start, "end": w.end } for w in segment.words or [] ]
            }
            whisper_results["segments"].append(segment_dict)

        start = time.time()
        model, meta, device = Tts.get_align_model_and_meta_and_device()
        result_aligned = whisperx.align(whisper_results["segments"], model, meta, sound.data, device=device)
        print("xxx", f"elapsed {(time.time() - start):.2f}")

        # --- Before and After Debugging ---
        print("--- Timestamp Comparison ---")
        print(f"{'Word':<20} | {'Before (faster-whisper)':<25} | {'After (whisperx)':<25}")
        print("-" * 75)

        original_words = [word for segment in segments_list for word in segment.words or []]
        aligned_words_list = [ word_info for segment in result_aligned["segments"] for word_info in segment.get("words", []) ]

        for i in range(min(len(original_words), len(aligned_words_list))):
            original_word = original_words[i]
            aligned_word_info = aligned_words_list[i]
            before_ts = f"{original_word.start:.2f} -> {original_word.end:.2f}"
            after_ts = f"{aligned_word_info.get('start', 'N/A'):.2f} -> {aligned_word_info.get('end', 'N/A'):.2f}"
            print(f"{original_word.word:<20} | {before_ts:<25} | {after_ts:<25}")

        # Create a new list of Word objects with updated timings
        updated_words: list[Word] = []
        for segment in result_aligned["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    # Create a new Word object with the updated timestamps
                    updated_word = ConcreteWord(
                        start=word_info.get('start'),
                        end=word_info.get('end'),
                        word=word_info['word'],
                        probability=0.0 # whisperx does not provide word-level probability in the same way
                    )
                    updated_words.append(updated_word)

        return updated_words

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
