"""
The step 3 choices: which project voice each roster character gets, stored as
`[project]/speaker_assign/voices.json` (see SpeakerAssignFileUtil).

On disk:

    {
        "format": "speaker_voices.v1",
        "voices": {"BARON": 3, "HELENE": 1}     # roster id -> voice index
    }

Voice indices are 0-based (the UI shows index + 1). Characters without an
entry have no voice chosen. Instances are immutable.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


VOICE_MAP_FILE_FORMAT = "speaker_voices.v1"


@dataclass(frozen=True)
class VoiceMapFile:
    voices: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for character_id, index in self.voices.items():
            if not isinstance(character_id, str) or not character_id:
                raise ValueError(f"Voice map needs character ids: {character_id!r}")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f"Voice map needs non-negative voice indices: {character_id}={index!r}")
        object.__setattr__(self, "voices", dict(self.voices))

    def get(self, character_id: str) -> int | None:
        return self.voices.get(character_id)

    def with_voice(self, character_id: str, voice_index: int | None) -> VoiceMapFile:
        """A copy with the character's voice set (None clears it)."""
        voices = {key: value for key, value in self.voices.items() if key != character_id}
        if voice_index is not None:
            voices[character_id] = voice_index
        return VoiceMapFile(voices)

    @staticmethod
    def all_set_to(character_ids: Iterable[str], voice_index: int | None) -> VoiceMapFile:
        """Every listed character with the same voice (None: no voices at all)."""
        if voice_index is None:
            return VoiceMapFile()
        return VoiceMapFile({character_id: voice_index for character_id in character_ids})

    def to_dict(self) -> dict[str, Any]:
        return {"format": VOICE_MAP_FILE_FORMAT, "voices": dict(self.voices)}

    @staticmethod
    def from_dict(value: Any) -> VoiceMapFile:
        if (not isinstance(value, dict) or value.get("format") != VOICE_MAP_FILE_FORMAT
                or not isinstance(value.get("voices"), dict)):
            raise ValueError(f"Not a {VOICE_MAP_FILE_FORMAT} voice map file")
        return VoiceMapFile(value["voices"])
