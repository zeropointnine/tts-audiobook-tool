from unittest.mock import patch

import numpy as np
import pytest

from tts_audiobook_tool.app_types import ConcreteWord, Sound, Word
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings
from tts_audiobook_tool.app_types.validation_result import TrimmedResult, WordErrorResult
from tts_audiobook_tool.text_ops.whitelist import Whitelist
from tts_audiobook_tool.validator import Validator


def _sound(duration: float) -> Sound:
    sr = 100
    rng = np.random.default_rng(0)
    return Sound(rng.uniform(-0.5, 0.5, round(duration * sr)).astype(np.float32), sr)


def _words(*items: tuple[str, float, float]) -> list[Word]:
    return [ConcreteWord(start=s, end=e, word=w, probability=1.0) for w, s, e in items]


@pytest.fixture(autouse=True)
def _english_whitelist():
    Whitelist().set_language_code("en")


def _run(words: list[Word], duration: float, source: str):
    sound = _sound(duration)
    wer = WordErrorResult(
        sound=sound, transcript_words=words, num_words=len(source.split()), threshold=0,
        findings=ValidationFindings(transcript_errors=["x"]),
    )
    # Identity local-minima and no silence-trim so timings are predictable
    with patch("tts_audiobook_tool.validator.SoundExtraUtil.get_local_minima", side_effect=lambda s, t: t), \
         patch("tts_audiobook_tool.validator.SilenceUtil.trim_silence_ends", side_effect=lambda s: (s, 0.0, s.duration)):
        return Validator.make_trimmed_result(wer, source, words, "en")


SRC = "the dog ran home today"


def _core(t0: float) -> list[tuple[str, float, float]]:
    return [(w, t0 + i * 0.4, t0 + i * 0.4 + 0.3) for i, w in enumerate(SRC.split())]  # spans t0..t0+1.9


def test_both_ends_trimmed_and_words_rebased_on_copies() -> None:
    """
    Happy path: two junk words before and two after the matching sentence, both well
    over the minimum trim length. Illustrates that:
    - both start_time and end_time are set,
    - only the matched words are kept,
    - their timings are re-based to the trimmed sound (first word starts near 0),
    - the caller's original Word objects are NOT mutated (copies are adjusted).
    """
    words = _words(("banana", 0.1, 0.4), ("pear", 0.5, 0.9), *_core(1.0), ("plum", 3.2, 3.5), ("kiwi", 3.6, 3.9))
    original = [(w.start, w.end) for w in words]
    result = _run(words, 4.0, SRC)
    assert isinstance(result, TrimmedResult)
    assert result.start_time is not None and result.end_time is not None
    assert [(w.start, w.end) for w in words] == original  # originals untouched
    assert [w.word for w in result.transcript_words] == SRC.split()
    assert 0.0 <= result.transcript_words[0].start < 0.1


def test_small_start_trim_is_skipped() -> None:
    """
    Leading junk lasts only ~0.06s (< MIN_SEMANTIC_TRIM_SECONDS) while trailing junk is
    long. Illustrates that the negligible start trim is dropped (start_time None) but
    the end trim is still applied.
    """
    words = _words(("banana", 0.0, 0.02), ("pear", 0.03, 0.05), *_core(0.06), ("plum", 3.2, 3.5), ("kiwi", 3.6, 3.9))
    result = _run(words, 4.0, SRC)
    assert isinstance(result, TrimmedResult)
    assert result.start_time is None
    assert result.end_time is not None


def test_small_end_trim_is_skipped() -> None:
    """
    Mirror of the small-start case: trailing junk is only ~0.05s while leading junk is
    long. Illustrates that the negligible end trim is dropped (end_time None) but
    the start trim is still applied.
    """
    words = _words(("banana", 0.1, 0.4), ("pear", 0.5, 0.9), *_core(1.0), ("plum", 2.92, 2.94), ("kiwi", 2.95, 2.97))
    result = _run(words, 3.0, SRC)
    assert isinstance(result, TrimmedResult)
    assert result.end_time is None
    assert result.start_time is not None


def test_no_result_when_both_trims_negligible() -> None:
    """
    Both the leading and trailing junk are shorter than the minimum. Illustrates that
    nothing is left to trim, so make_trimmed_result returns None (caller falls back
    to the plain WordErrorResult) rather than a no-op TrimmedResult.
    """
    words = _words(("banana", 0.0, 0.02), ("pear", 0.03, 0.05), *_core(0.06), ("plum", 1.96, 1.98), ("kiwi", 1.99, 2.0))
    assert _run(words, 2.0, SRC) is None


def test_ui_message_handles_zero_start() -> None:
    """
    Regression for truthiness checks on start/end: a start_time of 0.0 is a real
    value and must be reported as such, while a start_time of None must omit the
    "0s to ..." segment entirely.
    """
    r = TrimmedResult(_sound(1.0), [], 0.0, 4.16, original_duration=5.0)
    assert "0s to 0.00s" in r.get_ui_message()
    r = TrimmedResult(_sound(1.0), [], None, 4.16, original_duration=5.0)
    assert "0s to" not in r.get_ui_message()
