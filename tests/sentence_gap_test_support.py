"""Synthetic speech fixtures for sentence-gap tests.

``Utterance`` is shared by test_sentence_gap_util, test_concat_util and
test_real_time_playback_sentence_gaps.
"""

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase

SR = 24_000


def make_speech(duration: float, seed: int = 0) -> np.ndarray:
    """ Loud, amplitude-modulated tone: no internal valley deeper than ~8 dB """
    rng = np.random.default_rng(seed)
    n = round(duration * SR)
    t = np.arange(n, dtype=np.float64) / SR
    x = 0.2 * np.sin(2 * np.pi * 180 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t))
    x += 0.005 * rng.standard_normal(n)
    return x.astype(np.float32)


def make_floor(duration: float, level: float = 0.0, seed: int = 1) -> np.ndarray:
    """ Silence; `level` > 0 gives a noise floor of that std instead of digital zero """
    n = round(duration * SR)
    if level == 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(n)).astype(np.float32)


class Utterance:
    """
    Two sentences, "Hello there friend." and "How are you today", built from loud
    "words" with short soft dips between them and a controllable gap between the
    sentences. Holds both the true timings and the (possibly perturbed) timings a
    Whisper-like transcriber would have reported.
    """

    def __init__(self, gap_s: float, floor_level: float = 0.0,
                 whisper_end_shift: float = 0.0, whisper_start_shift: float = 0.0,
                 word_dip_s: float = 0.025):
        s1 = [("Hello", 0.30), ("there", 0.30), ("friend.", 0.40)]
        s2 = [("How", 0.25), ("are", 0.25), ("you", 0.25), ("today", 0.40)]
        parts: list[np.ndarray] = []
        self.words: list[dict] = []
        t = 0.0

        def add_sentence(sentence: list[tuple[str, float]], lead_in: float) -> None:
            nonlocal t
            for k, (word, duration) in enumerate(sentence):
                parts.append(make_speech(duration, seed=len(self.words)))
                self.words.append({"word": f" {word}", "start": t, "end": t + duration})
                t += duration
                if k < len(sentence) - 1:
                    # soft dip between words (-30 dB-ish), not silence
                    parts.append(make_floor(word_dip_s, level=0.004, seed=len(self.words)))
                    t += word_dip_s

        add_sentence(s1, 0.0)
        self.true_gap_start = t
        parts.append(make_floor(gap_s, level=floor_level))
        t += gap_s
        self.true_gap_end = t
        add_sentence(s2, gap_s)

        self.sound = Sound(np.concatenate(parts), SR)

        # What "Whisper" said: shifts applied to the words flanking the boundary
        self.words[2] = {**self.words[2], "end": self.words[2]["end"] + whisper_end_shift}
        self.words[3] = {**self.words[3], "start": self.words[3]["start"] + whisper_start_shift}

        self.timed_phrases = [
            TimedPhrase("Hello there friend. ", 0.0, self.words[3]["start"], reason=Reason.SENTENCE),
            TimedPhrase("How are you today", self.words[3]["start"], self.sound.duration, reason=Reason.PARAGRAPH),
        ]
