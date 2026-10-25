"""
`Voice`: one entry of a project's ordered `voice_references` list (project
spec v5; see docs-dev/project-spec-v5.md).

On disk an entry is a flat JSON object:

    {
        "name": "Narrator",                 # optional user-facing name (one word); written only when non-empty
        "file_name": "belle.flac",          # project-local voice sample
        "transcript": "...",                # "" when unused/absent
        "crop_file_name": "<hex>.flac",     # optional crop group (all four or none)
        "crop_start": 1.0,
        "crop_end": 3.5,
        "crop_transcript": "...",
        "room_reverb": "off",               # RoomReverb id, always written
        "pan": -0.3,                        # optional stereo position, -1 (left) to 1 (right);
                                            # written only when not centered
        "presets": {"<model id>": "alba"},  # optional per-TTS-model preset voice ids (eg a Pocket
                                            # predefined voice, a Qwen3-TTS CustomVoice speaker);
                                            # written only when non-empty
        "instructions": {"<model id>": "..."}  # optional per-TTS-model voice instructions for
                                            # "instruct" models; written only when non-empty
    }

In memory the four crop keys are one optional `VoiceCrop`. Instances are
immutable; edits make a new instance with `dataclasses.replace`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from tts_audiobook_tool.app_types import RoomReverb

# Managed crop files live in this subdirectory of the project's voice dir.
VOICE_CROP_SUBDIR = "crops"

CROP_FILE_NAME_FIELD = "crop_file_name"
CROP_START_FIELD = "crop_start"
CROP_END_FIELD = "crop_end"
CROP_TRANSCRIPT_FIELD = "crop_transcript"
CROP_FIELDS = (CROP_FILE_NAME_FIELD, CROP_START_FIELD, CROP_END_FIELD, CROP_TRANSCRIPT_FIELD)
ROOM_REVERB_FIELD = "room_reverb"
PAN_FIELD = "pan"
PRESETS_FIELD = "presets"
INSTRUCTIONS_FIELD = "instructions"
NAME_FIELD = "name"


def stereo_position_to_percent(pan: float) -> int:
    """The UI's -100 (full left) to 100 (full right) value of a stored pan."""
    return int(round(pan * 100))


def stereo_position_label(pan: float) -> str:
    """eg "Center", "Left 30", "Right 30"."""
    percent = stereo_position_to_percent(pan)
    if percent == 0:
        return "Center"
    return f"{'Left' if percent < 0 else 'Right'} {abs(percent)}"


def normalize_crop_file_name(value: Any) -> str | None:
    """Accept only a FLAC basename; the managed directory is not persisted."""
    if not isinstance(value, str):
        return None
    if (not value.endswith(".flac") or value == ".flac"
            or any(char in value for char in ("/", "\\", ":", "\x00"))):
        return None
    return value


@dataclass(frozen=True)
class VoiceCrop:
    """An active trim of a voice sample, materialized as `voice/crops/<file_name>`."""

    file_name: str
    start: float
    end: float
    transcript: str = ""

    def __post_init__(self) -> None:
        if normalize_crop_file_name(self.file_name) is None:
            raise ValueError("trim requires an explicit <name>.flac basename")
        if not isinstance(self.transcript, str):
            raise ValueError("trim transcript must be a string")
        start, end = float(self.start), float(self.end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("expected finite seconds with 0 <= start < end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def relative_path(self) -> str:
        """Voice-dir-relative path of the crop file."""
        return f"{VOICE_CROP_SUBDIR}/{self.file_name}"


@dataclass(frozen=True)
class Voice:
    """A project voice: optional user-facing `name` (a convenience name only;
    never affects generation), reference sample, its transcript, optional crop, room reverb,
    stereo position (`pan`, -1 full left to 1 full right, 0 centered), model
    preset voice ids (`presets`, keyed by TTS model id, for models with
    built-in voices), and voice `instructions` keyed by TTS model id (for
    "instruct" models). Empty strings / empty maps mean unused; an empty
    `file_name` means the voice has no sample. `Voice()` is the default voice."""

    file_name: str = ""
    transcript: str = ""
    crop: VoiceCrop | None = None
    room_reverb: RoomReverb = RoomReverb.OFF
    pan: float = 0.0
    # TTS model id -> preset voice id. Treat as read-only; excluded from hashing.
    presets: dict[str, str] = field(default_factory=dict, hash=False)
    # TTS model id -> instructions. Treat as read-only; excluded from hashing.
    instructions: dict[str, str] = field(default_factory=dict, hash=False)
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        pan = float(self.pan)
        if not (math.isfinite(pan) and -1.0 <= pan <= 1.0):
            raise ValueError("pan must be finite and between -1 and 1")
        object.__setattr__(self, "pan", pan)
        for name in ("presets", "instructions"):
            value = getattr(self, name)
            if not isinstance(value, dict) or not all(
                    isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
                raise ValueError(f"{name} must be a dict of model id to string")
            # Own a copy so the caller's dict cannot mutate this instance.
            object.__setattr__(self, name, dict(value))

    @property
    def has_sample(self) -> bool:
        return bool(self.file_name)

    @property
    def is_default(self) -> bool:
        """Whether every value is the default (a fresh `Voice()`)."""
        return self == Voice()

    def preset_for(self, model_id: str) -> str:
        """This voice's preset voice id for a TTS model, or ""."""
        return self.presets.get(model_id, "")

    @property
    def crop_range(self) -> tuple[float, float] | None:
        return (self.crop.start, self.crop.end) if self.crop is not None else None

    def to_dict(self) -> dict[str, Any]:
        """The on-disk form. `room_reverb` is always written; `pan` only when nonzero;
        `name`, `presets` and `instructions` only when non-empty."""
        result: dict[str, Any] = {}
        if self.name:
            result[NAME_FIELD] = self.name
        result["file_name"] = self.file_name
        result["transcript"] = self.transcript
        if self.crop is not None:
            result[CROP_FILE_NAME_FIELD] = self.crop.file_name
            result[CROP_START_FIELD] = self.crop.start
            result[CROP_END_FIELD] = self.crop.end
            result[CROP_TRANSCRIPT_FIELD] = self.crop.transcript
        result[ROOM_REVERB_FIELD] = self.room_reverb.id
        if self.pan != 0:
            result[PAN_FIELD] = self.pan
        if self.presets:
            result[PRESETS_FIELD] = dict(self.presets)
        if self.instructions:
            result[INSTRUCTIONS_FIELD] = dict(self.instructions)
        return result

    @staticmethod
    def from_dict(
            raw: Any,
            where: str = "voice_references",
            index: int = 0,
            *,
            default_room_reverb: RoomReverb = RoomReverb.OFF,
    ) -> Voice | None:
        """Parse one stored entry.

        Raises ValueError for a structurally invalid entry. An empty
        `file_name` is a voice without a sample. Returns None when a non-empty
        `file_name` cannot be a project-local path (the entry is dropped, with
        a warning). Malformed crop metadata drops only the crop; an unknown
        `room_reverb` id loads as Off. A missing `room_reverb` key (all
        pre-v5 data) takes `default_room_reverb`. A missing or non-numeric
        `pan` loads as centered; an out-of-range one is clamped. A missing or
        non-object `presets` / `instructions` (including a legacy plain string)
        loads as an empty map. A missing or non-string `name` loads as "".
        """
        from tts_audiobook_tool.app_support import path_norm
        from tts_audiobook_tool.l import L

        if (not isinstance(raw, dict) or not isinstance(raw.get("file_name"), str)
                or not isinstance(raw.get("transcript", ""), str)):
            raise ValueError(f"{where}[{index}]: expected file_name and optional transcript strings")
        file_name, _ = path_norm.normalize_stored_relative_path(raw["file_name"])
        if raw["file_name"] and not file_name:
            # A stored name that cannot be a project-local file (e.g. a stray
            # absolute path) drops that entry rather than failing the project.
            L.w(f"{where}[{index}].file_name is not a usable project-local path; dropping entry")
            return None

        room_reverb = default_room_reverb
        if ROOM_REVERB_FIELD in raw:
            value = raw[ROOM_REVERB_FIELD]
            room_reverb = value if isinstance(value, RoomReverb) else (
                RoomReverb.get_by_id(value) if isinstance(value, str) else None
            ) or RoomReverb.OFF

        return Voice(
            file_name=file_name,
            transcript=raw.get("transcript", ""),
            crop=_parse_crop(raw, where, index),
            room_reverb=room_reverb,
            pan=_parse_pan(raw, where, index),
            presets=_parse_str_map(raw, PRESETS_FIELD, where, index),
            instructions=_parse_str_map(raw, INSTRUCTIONS_FIELD, where, index),
            name=_parse_name(raw, where, index),
        )


def ensure_voice_slot(voices: list[Voice]) -> list[Voice]:
    """A project's voice list is never empty: an empty list becomes one default voice."""
    return voices if voices else [Voice()]


def _parse_str_map(raw: dict, field: str, where: str, index: int) -> dict[str, str]:
    """A model id -> string map. Empty when absent or not a map (the latter
    with a warning; a legacy string value is not migrated). Entries with a
    non-string key or value are dropped."""
    from tts_audiobook_tool.l import L

    value = raw.get(field, {})
    if not isinstance(value, dict):
        L.w(f"{where}[{index}]: invalid {field} ({value!r}); using empty")
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}


def _parse_name(raw: dict, where: str, index: int) -> str:
    """"" when absent or not a string (the latter with a warning)."""
    from tts_audiobook_tool.l import L

    value = raw.get(NAME_FIELD, "")
    if not isinstance(value, str):
        L.w(f"{where}[{index}]: invalid {NAME_FIELD} ({value!r}); using empty")
        return ""
    return value


def _parse_pan(raw: dict, where: str, index: int) -> float:
    """Centered when absent or malformed (with a warning); clamped to [-1, 1]."""
    from tts_audiobook_tool.l import L

    if PAN_FIELD not in raw:
        return 0.0
    value = raw[PAN_FIELD]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        L.w(f"{where}[{index}]: invalid pan ({value!r}); using center")
        return 0.0
    return min(max(float(value), -1.0), 1.0)


def _parse_crop(raw: dict, where: str, index: int) -> VoiceCrop | None:
    """Drop malformed crop metadata with a warning; never infer a crop path.

    Times are accepted as numbers or (pre-v5) numeric strings. Original
    sample files and transcripts are retained untouched.
    """
    from tts_audiobook_tool.l import L

    if not any(name in raw for name in CROP_FIELDS):
        return None
    start = raw.get(CROP_START_FIELD)
    end = raw.get(CROP_END_FIELD)
    try:
        if start is None or end is None:
            raise ValueError("trim start and end must both be present")
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("trim start and end must be numbers")
        return VoiceCrop(
            file_name=raw.get(CROP_FILE_NAME_FIELD),  # type: ignore[arg-type]
            start=float(start),
            end=float(end),
            transcript=raw.get(CROP_TRANSCRIPT_FIELD, ""),
        )
    except (TypeError, ValueError, OverflowError) as e:
        L.w(f"{where}[{index}]: invalid trim ({e}); dropping trim")
        return None
