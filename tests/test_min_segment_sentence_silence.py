"""
`MinSegmentSentenceSilence` enum and the project attribute that stores it.
"""

from typing import Any

from tts_audiobook_tool.app_types import MinSegmentSentenceSilence
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil


def test_enum_ids_and_durations() -> None:
    # Each member pairs a JSON id with a duration in seconds
    assert MinSegmentSentenceSilence.DISABLED.id == "disabled"
    assert MinSegmentSentenceSilence.DISABLED.duration == 0.0
    assert MinSegmentSentenceSilence.ENABLED.id == "enabled"
    assert MinSegmentSentenceSilence.ENABLED.duration == 0.5
    assert MinSegmentSentenceSilence.DISABLED.label == "disabled"
    assert MinSegmentSentenceSilence.ENABLED.label == "enabled"
    assert MinSegmentSentenceSilence.get_default() is MinSegmentSentenceSilence.DISABLED


def test_get_by_id() -> None:
    assert MinSegmentSentenceSilence.get_by_id("enabled") is MinSegmentSentenceSilence.ENABLED
    assert MinSegmentSentenceSilence.get_by_id("bogus") is None


def test_new_project_defaults_to_disabled() -> None:
    assert Project().min_segment_sentence_silence is MinSegmentSentenceSilence.DISABLED


def test_missing_key_normalizes_to_disabled_without_warning() -> None:
    # Projects saved before the attribute existed must load silently
    warnings: list[str] = []
    d: dict[str, Any] = {}
    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=warnings)
    assert d["min_segment_sentence_silence"] is MinSegmentSentenceSilence.DISABLED
    assert not [w for w in warnings if "min_segment_sentence_silence" in w]


def test_unknown_value_falls_back_to_disabled() -> None:
    d: dict[str, Any] = {"min_segment_sentence_silence": "bogus"}
    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=[])
    assert d["min_segment_sentence_silence"] is MinSegmentSentenceSilence.DISABLED


def test_round_trips_through_json_dict() -> None:
    project = Project()
    project.min_segment_sentence_silence = MinSegmentSentenceSilence.ENABLED
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved["min_segment_sentence_silence"] == "enabled"

    loaded = Project.model_validate(saved)
    assert loaded.min_segment_sentence_silence is MinSegmentSentenceSilence.ENABLED
