"""`Voice.presets` / `Voice.instructions` storage."""
import pytest

from tts_audiobook_tool.app_types.voice import Voice


def test_empty_fields_are_not_written_and_set_fields_round_trip():
    data = Voice("a.flac").to_dict()
    assert "presets" not in data and "instructions" not in data
    voice = Voice("a.flac", presets={"pocket_local": "alba", "qwen3tts_local": "Ryan"},
                  instructions={"qwen3tts_local": "Speak softly."})
    data = voice.to_dict()
    assert data["presets"] == {"pocket_local": "alba", "qwen3tts_local": "Ryan"}
    assert data["instructions"] == {"qwen3tts_local": "Speak softly."}
    assert Voice.from_dict(data) == voice
    assert voice.preset_for("pocket_local") == "alba" and voice.preset_for("other") == ""


@pytest.mark.parametrize("stored", [None, 3, ["x"], "Speak softly."])
def test_missing_or_malformed_fields_load_empty(stored):
    assert Voice.from_dict({"file_name": "a.flac"}) == Voice("a.flac")
    voice = Voice.from_dict({"file_name": "a.flac", "presets": stored, "instructions": stored})
    assert voice is not None and voice.presets == {} and voice.instructions == {}


def test_retired_single_preset_key_is_ignored():
    voice = Voice.from_dict({"file_name": "a.flac", "preset": "alba"})
    assert voice == Voice("a.flac")
    assert "preset" not in voice.to_dict()


@pytest.mark.parametrize("field", ["presets", "instructions"])
def test_maps_drop_non_string_entries(field):
    voice = Voice.from_dict({"file_name": "a.flac", field: {"a": "x", "b": 3, "c": None}})
    assert voice is not None and getattr(voice, field) == {"a": "x"}


@pytest.mark.parametrize("field", ["presets", "instructions"])
def test_voice_does_not_alias_caller_dict_and_stays_hashable(field):
    source = {"a": "x"}
    voice = Voice("a.flac", **{field: source})
    source["a"] = "changed"
    assert getattr(voice, field) == {"a": "x"}
    hash(voice)


@pytest.mark.parametrize("field, value", [
    ("presets", None), ("presets", "alba"), ("presets", {"a": 1}),
    ("instructions", None), ("instructions", "x"), ("instructions", {"a": 1}),
])
def test_voice_rejects_wrong_types(field, value):
    with pytest.raises(ValueError):
        Voice("a.flac", **{field: value})
