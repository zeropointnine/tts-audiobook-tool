"""`Voice.preset` / `Voice.instructions` storage."""
import pytest

from tts_audiobook_tool.app_types.voice import Voice


def test_empty_fields_are_not_written_and_set_fields_round_trip():
    data = Voice("a.flac").to_dict()
    assert "preset" not in data and "instructions" not in data
    voice = Voice("a.flac", preset="alba", instructions={"qwen3tts_local": "Speak softly."})
    data = voice.to_dict()
    assert data["preset"] == "alba" and data["instructions"] == {"qwen3tts_local": "Speak softly."}
    assert Voice.from_dict(data) == voice


@pytest.mark.parametrize("stored", [None, 3, ["x"], "Speak softly."])
def test_missing_or_malformed_fields_load_empty(stored):
    assert Voice.from_dict({"file_name": "a.flac"}) == Voice("a.flac")
    voice = Voice.from_dict({"file_name": "a.flac", "preset": stored if not isinstance(stored, str) else 3,
                             "instructions": stored})
    assert voice is not None and voice.preset == "" and voice.instructions == {}


def test_instructions_drop_non_string_entries():
    voice = Voice.from_dict({"file_name": "a.flac", "instructions": {"a": "x", "b": 3, "c": None}})
    assert voice is not None and voice.instructions == {"a": "x"}


def test_voice_does_not_alias_caller_dict_and_stays_hashable():
    source = {"a": "x"}
    voice = Voice("a.flac", instructions=source)
    source["a"] = "changed"
    assert voice.instructions == {"a": "x"}
    hash(voice)


@pytest.mark.parametrize("field, value", [
    ("preset", None), ("instructions", None), ("instructions", "x"), ("instructions", {"a": 1}),
])
def test_voice_rejects_wrong_types(field, value):
    with pytest.raises(ValueError):
        Voice("a.flac", **{field: value})
