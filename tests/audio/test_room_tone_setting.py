"""
The `room_tone` project attribute: default, load normalization (including the
legacy bool `fill_room_tone`), and persistence.
"""

from typing import Any

from tts_audiobook_tool.app_types import RoomTone
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil


def normalized(d: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=warnings if warnings is not None else [])
    return d


def test_new_project_defaults_to_off() -> None:
    assert RoomTone.get_default() is RoomTone.OFF
    assert Project().room_tone is RoomTone.OFF


def test_missing_key_normalizes_to_default_without_warning() -> None:
    # Projects saved before the attribute existed must load silently
    warnings: list[str] = []
    d = normalized({}, warnings)
    assert d["room_tone"] is RoomTone.OFF
    assert not [w for w in warnings if "room_tone" in w]


def test_unknown_id_falls_back_to_default() -> None:
    assert normalized({"room_tone": "loud"})["room_tone"] is RoomTone.OFF


def test_legacy_bool_true_maps_to_high_and_is_dropped() -> None:
    # The old on/off setting filled at -65 dBFS, which is HIGH now
    d = normalized({"fill_room_tone": True})
    assert d["room_tone"] is RoomTone.HIGH
    assert "fill_room_tone" not in d
    assert RoomTone.HIGH.level_db == -65.0


def test_legacy_bool_false_maps_to_off() -> None:
    assert normalized({"fill_room_tone": False})["room_tone"] is RoomTone.OFF


def test_new_key_wins_over_legacy_bool() -> None:
    d = normalized({"fill_room_tone": True, "room_tone": "low"})
    assert d["room_tone"] is RoomTone.LOW


def test_round_trips_through_json_dict() -> None:
    project = Project()
    project.room_tone = RoomTone.MEDIUM
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["room_tone"] == "medium"
    assert "fill_room_tone" not in saved

    # Project validation normalizes the dict itself
    loaded = Project.model_validate(saved)
    assert loaded.room_tone is RoomTone.MEDIUM
