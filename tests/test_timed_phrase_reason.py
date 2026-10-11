import json

from tts_audiobook_tool.app_types.app_metadata import app_metadata_text_segments_to_dicts
from tts_audiobook_tool.app_types.force_align_util import ForceAlignUtil
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase


class FakeWord:
    def __init__(self, word: str, start: float, end: float):
        self.word, self.start, self.end, self.probability = word, start, end, 1.0


def test_reason_is_written_to_the_dict_only_on_request() -> None:
    # The sidecar asks for it; everything else (notably the ABR app metadata, whose
    # format is fixed) gets the same three keys as before.
    item = TimedPhrase("Hello. ", 0.0, 1.0, reason=Reason.SENTENCE)

    assert TimedPhrase.to_dict(item) == {"text": "Hello. ", "time_start": 0.0, "time_end": 1.0}
    assert TimedPhrase.to_dict(item, include_reason=True)["reason"] == "s"
    assert app_metadata_text_segments_to_dicts([item]) == [{"text": "Hello. ", "time_start": 0.0, "time_end": 1.0}]
    assert app_metadata_text_segments_to_dicts([[item, item]])[0][0].keys() == {"text", "time_start", "time_end"}


def test_unknown_reason_writes_no_key() -> None:
    item = TimedPhrase("Hello. ", 0.0, 1.0)

    assert item.reason is None
    assert "reason" not in TimedPhrase.to_dict(item, include_reason=True)


def test_every_reason_round_trips_through_json() -> None:
    items = [TimedPhrase(f"t{r.name} ", float(i), float(i + 1), reason=r) for i, r in enumerate(Reason)]

    dicts = json.loads(json.dumps(TimedPhrase.timed_phrases_to_dicts(items, include_reason=True)))
    loaded = TimedPhrase.dicts_to_timed_phrases(dicts)

    assert not isinstance(loaded, str)
    assert [t.reason for t in loaded] == list(Reason)


def test_legacy_dicts_without_reason_still_load() -> None:
    # Sidecars written before reasons were recorded.
    loaded = TimedPhrase.dicts_to_timed_phrases([{"text": "Old. ", "time_start": 0.0, "time_end": 2.0}])

    assert not isinstance(loaded, str)
    assert loaded[0].reason is None


def test_make_using_carries_the_phrases_reason() -> None:
    item = TimedPhrase.make_using(Phrase("One. ", Reason.PARAGRAPH), 0.0, 1.0)

    assert item.reason is Reason.PARAGRAPH


def test_force_align_gives_each_timed_phrase_its_source_reason() -> None:
    # The sidecar's timed phrases come from here; each must carry the Reason of the
    # source phrase it was aligned to.
    phrases = [
        Phrase("Hello there, ", Reason.PHRASE),
        Phrase("friend. ", Reason.SENTENCE),
        Phrase("How are you", Reason.PARAGRAPH),
    ]
    words = [FakeWord(w, i * 0.3, i * 0.3 + 0.3) for i, w in enumerate("Hello there, friend. How are you".split())]

    timed = ForceAlignUtil.make_timed_phrases(phrases, words, sound_duration=2.0)  # type: ignore[arg-type]

    assert [t.reason for t in timed] == [Reason.PHRASE, Reason.SENTENCE, Reason.PARAGRAPH]


def test_force_align_single_phrase_keeps_its_reason() -> None:
    timed = ForceAlignUtil.make_timed_phrases([Phrase("Only. ", Reason.SECTION_BREAK)], [], sound_duration=1.0)

    assert timed[0].reason is Reason.SECTION_BREAK


def test_timed_phrase_unpickled_from_before_reasons_existed_has_none() -> None:
    # Enhance artifacts pickle TimedPhrase objects; ones saved before the attribute
    # existed have no 'reason' in their __dict__ and must still read as unknown.
    import pickle
    item = TimedPhrase("Old. ", 0.0, 1.0)
    del item.__dict__["reason"]

    restored = pickle.loads(pickle.dumps(item))

    assert restored.reason is None


def test_saved_sidecar_file_records_a_reason_for_every_timed_phrase(tmp_path) -> None:
    # End to end through the one real sidecar writer (GenerateUtil.save_sound_and_timing_json
    # -> from_validation_result -> SegmentTranscriptUtil.save): the JSON on disk must carry
    # the source phrase's reason on every timed phrase, including the last one.
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np

    from tts_audiobook_tool.app_types import ConcreteWord, Sound, Strictness
    from tts_audiobook_tool.app_types.phrase import PhraseGroup
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
    from tts_audiobook_tool.validator import Validator

    project = Project.model_validate({"language_code": "en"})
    group = PhraseGroup([
        Phrase("Hello there, ", Reason.PHRASE),
        Phrase("friend. ", Reason.SENTENCE),
        Phrase("How are you? ", Reason.PARAGRAPH),
    ])
    sr = 24000
    sound = Sound(np.zeros(sr * 3, dtype=np.float32), sr)
    words = [ConcreteWord(i * 0.4, i * 0.4 + 0.3, w, 1.0) for i, w in enumerate(["Hello", "there,", "friend.", "How", "are", "you?"])]
    with patch("tts_audiobook_tool.validator.ModelManager.has_yamnet_detector", return_value=False):
        result = Validator.validate(sound, group.text, words, "en", Strictness.MODERATE)

    err, sound_path = GenerateUtil.save_sound_and_timing_json(
        state=SimpleNamespace(project=project),  # type: ignore[arg-type]
        phrase_group=group, index=0, validation_result=result, is_real_time=False,
        voice_tag="v", dir_path=str(tmp_path),
    )

    assert err == ""
    sidecar = tmp_path / (sound_path.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".json")
    on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
    assert [p["reason"] for p in on_disk["timed_phrases"]] == ["is", "s", "p"]
    loaded = SegmentTranscriptUtil.load(sidecar)
    assert not isinstance(loaded, str)
    assert [t.reason for t in loaded.timed_phrases] == [Reason.PHRASE, Reason.SENTENCE, Reason.PARAGRAPH]
