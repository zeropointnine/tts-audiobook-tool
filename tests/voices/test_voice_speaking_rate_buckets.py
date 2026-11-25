"""Per-voice speaking-rate buckets: strict construction and tolerant persistence."""
import json
from pathlib import Path

import pytest

from tts_audiobook_tool.app_types.voice import (
    SPEAKING_RATE_BUCKET_MAX,
    SPEAKING_RATE_BUCKET_MIN,
    Voice,
)
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project

MODEL = "zonos3_mini_sglomni"
FUTURE_MODEL = "future/model:v2"


def test_boundaries_and_exact_unknown_model_ids_round_trip():
    # Storage preserves model ids verbatim, without requiring a catalog entry.
    assert (SPEAKING_RATE_BUCKET_MIN, SPEAKING_RATE_BUCKET_MAX) == (1, 30)
    buckets = {MODEL: 1, FUTURE_MODEL: 30, MODEL.upper(): 15}
    voice = Voice(speaking_rate_buckets=buckets)
    assert voice.to_dict()["speaking_rate_buckets"] == buckets
    assert Voice.from_dict(voice.to_dict()) == voice
    assert voice.speaking_rate_buckets.get("missing") is None
    assert not voice.is_default


@pytest.mark.parametrize("buckets", [None, [], "15", 15, {1: 15}])
def test_constructor_rejects_invalid_maps_and_model_ids(buckets):
    with pytest.raises(ValueError, match="speaking_rate_buckets"):
        Voice(speaking_rate_buckets=buckets)


@pytest.mark.parametrize("bucket", [True, False, None, "15", 15.0, 0, 31, -1])
def test_constructor_rejects_noninteger_or_out_of_range_buckets(bucket):
    with pytest.raises(ValueError, match="speaking_rate_buckets"):
        Voice(speaking_rate_buckets={MODEL: bucket})


def test_map_ownership_is_defensive_and_voice_stays_hashable():
    # Both input maps and exported maps are separate from the voice's owned map.
    source = {MODEL: 15}
    voice = Voice(speaking_rate_buckets=source)
    source[MODEL] = 20
    exported = voice.to_dict()
    exported["speaking_rate_buckets"][MODEL] = 25
    assert voice.speaking_rate_buckets == {MODEL: 15}
    hash(voice)

    stored_buckets = {MODEL: 10}
    loaded = Voice.from_dict({"file_name": "", "speaking_rate_buckets": stored_buckets})
    stored_buckets.clear()
    assert loaded is not None and loaded.speaking_rate_buckets == {MODEL: 10}


def test_missing_and_empty_maps_are_disabled_and_omitted(monkeypatch):
    warnings = []
    monkeypatch.setattr(L, "w", warnings.append)
    default = Voice()
    explicit_empty = Voice(speaking_rate_buckets={})
    assert default.speaking_rate_buckets == {}
    assert default.speaking_rate_buckets is not explicit_empty.speaking_rate_buckets
    assert "speaking_rate_buckets" not in default.to_dict()
    for stored in ({"file_name": ""}, {"file_name": "", "speaking_rate_buckets": {}}):
        loaded = Voice.from_dict(stored)
        assert loaded is not None and loaded == default
        assert "speaking_rate_buckets" not in loaded.to_dict()
    assert warnings == []


@pytest.mark.parametrize("stored", [None, 15, True, [15], "15"])
def test_malformed_maps_load_disabled_with_warning(stored, monkeypatch):
    warnings = []
    monkeypatch.setattr(L, "w", warnings.append)
    voice = Voice.from_dict({"file_name": "", "speaking_rate_buckets": stored}, "voices", 2)
    assert voice is not None and voice.speaking_rate_buckets == {}
    assert "speaking_rate_buckets" not in voice.to_dict()
    assert len(warnings) == 1
    assert "voices[2]" in warnings[0] and "speaking_rate_buckets" in warnings[0]


def test_invalid_entries_are_dropped_not_clamped_and_valid_siblings_survive(monkeypatch):
    # Invalid overrides revert to Disabled independently; no enabled fallback is invented.
    warnings = []
    monkeypatch.setattr(L, "w", warnings.append)
    invalid = {"bool": True, "false": False, "float": 15.0, "string": "15",
               "none": None, "low": 0, "high": 31, 9: 15}
    valid = {MODEL: 1, FUTURE_MODEL: 30}
    voice = Voice.from_dict({"file_name": "", "speaking_rate_buckets": {**invalid, **valid}})
    assert voice is not None and voice.speaking_rate_buckets == valid
    assert voice.to_dict()["speaking_rate_buckets"] == valid
    assert len(warnings) == len(invalid)
    assert all("speaking_rate_buckets" in warning and "dropping" in warning for warning in warnings)


def test_project_save_and_reload_preserves_per_voice_per_model_overrides(tmp_path: Path):
    # Exercise the real Project serializer and JSON loader, including a disabled voice.
    voices = [
        Voice("a.flac", speaking_rate_buckets={MODEL: 1, FUTURE_MODEL: 30}),
        Voice("b.flac", speaking_rate_buckets={MODEL: 21, MODEL.upper(): 12}),
        Voice(),
    ]
    project = Project(dir_path=str(tmp_path), voice_references=voices)
    assert project.save() == ""
    saved_text = (tmp_path / "project.json").read_text(encoding="utf-8")
    saved = json.loads(saved_text)
    assert [entry.get("speaking_rate_buckets") for entry in saved["voice_references"]] == [
        {MODEL: 1, FUTURE_MODEL: 30}, {MODEL: 21, MODEL.upper(): 12}, None,
    ]
    loaded = Project.model_validate_json(saved_text)
    assert loaded.voice_references == voices
    assert loaded.voice_references[0].speaking_rate_buckets is not voices[0].speaking_rate_buckets
