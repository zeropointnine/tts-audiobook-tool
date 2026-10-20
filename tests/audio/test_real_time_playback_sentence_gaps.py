from types import SimpleNamespace
from typing import cast

from sentence_gap_test_support import Utterance

from tts_audiobook_tool import real_time_playback
from tts_audiobook_tool.app_types import MinSegmentSentenceSilence
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.validation_result import SkippedResult, TrimmedResult
from tts_audiobook_tool.state import State


def make_state(setting: MinSegmentSentenceSilence) -> State:
    project = SimpleNamespace(min_segment_sentence_silence=setting)
    return cast(State, SimpleNamespace(project=project))


def make_phrase_group() -> PhraseGroup:
    return PhraseGroup([
        Phrase("Hello there friend. ", Reason.SENTENCE),
        Phrase("How are you today", Reason.PARAGRAPH),
    ])


def make_transcript_result(u: Utterance) -> TrimmedResult:
    words = [SimpleNamespace(word=w["word"], start=w["start"], end=w["end"], probability=1.0) for w in u.words]
    return TrimmedResult(u.sound, transcript_words=words, start_time=0.0, end_time=None,  # type: ignore[arg-type]
                         original_duration=u.sound.duration)


def test_extends_short_sentence_gap_from_in_memory_transcript() -> None:
    # Realtime has no sidecar to read; the validation result's words drive the extension
    u = Utterance(gap_s=0.15)
    sound = real_time_playback.extend_sentence_gaps(
        make_state(MinSegmentSentenceSilence.ENABLED), make_phrase_group(), make_transcript_result(u)
    )
    assert sound.duration > u.sound.duration + 0.2


def test_no_extension_when_disabled_or_stt_skipped() -> None:
    # Setting off, or a segment generated before realtime started validating: audio unchanged
    u = Utterance(gap_s=0.15)
    disabled = real_time_playback.extend_sentence_gaps(
        make_state(MinSegmentSentenceSilence.DISABLED), make_phrase_group(), make_transcript_result(u)
    )
    skipped = real_time_playback.extend_sentence_gaps(
        make_state(MinSegmentSentenceSilence.ENABLED), make_phrase_group(), SkippedResult(u.sound, message="")
    )
    assert disabled is u.sound
    assert skipped is u.sound
