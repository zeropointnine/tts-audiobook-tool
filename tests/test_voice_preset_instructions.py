"""`Voice.preset` / `Voice.instructions` storage."""
import pytest

from tts_audiobook_tool.app_types.voice import Voice


def test_empty_fields_are_not_written_and_set_fields_round_trip():
    data = Voice("a.flac").to_dict()
    assert "preset" not in data and "instructions" not in data
    voice = Voice("a.flac", preset="alba", instructions="Speak softly.")
    data = voice.to_dict()
    assert data["preset"] == "alba" and data["instructions"] == "Speak softly."
    assert Voice.from_dict(data) == voice


@pytest.mark.parametrize("stored", [None, 3, ["x"]])
def test_missing_or_malformed_fields_load_empty(stored):
    assert Voice.from_dict({"file_name": "a.flac"}) == Voice("a.flac")
    voice = Voice.from_dict({"file_name": "a.flac", "preset": stored, "instructions": stored})
    assert voice is not None and voice.preset == "" and voice.instructions == ""


@pytest.mark.parametrize("field", ["preset", "instructions"])
def test_voice_rejects_non_string(field):
    with pytest.raises(ValueError):
        Voice("a.flac", **{field: None})
