import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.reason_pauses import ReasonPauses, ReasonPauseTypes


EXPECTED_PAUSES = {
    ReasonPauseTypes.NORMAL: {
        Reason.UNDEFINED: 1.0,
        Reason.WORD: 0.1,
        Reason.PHRASE: 0.5,
        Reason.SENTENCE: 0.9,
        Reason.PARAGRAPH: 1.2,
        Reason.SPACE_BREAK: 2.0,
        Reason.SECTION_BREAK: 2.5,
    },
    ReasonPauseTypes.SHORTER: {
        Reason.UNDEFINED: 1.0,
        Reason.WORD: 0.1,
        Reason.PHRASE: 0.3,
        Reason.SENTENCE: 0.6,
        Reason.PARAGRAPH: 0.9,
        Reason.SPACE_BREAK: 1.5,
        Reason.SECTION_BREAK: 2.0,
    },
}


@pytest.mark.parametrize("pause_type", list(ReasonPauseTypes))
def test_reason_pause_type_has_all_current_pause_durations(
    pause_type: ReasonPauseTypes,
) -> None:
    pauses = pause_type.value
    expected_pauses = EXPECTED_PAUSES[pause_type]

    assert isinstance(pauses, ReasonPauses)
    assert set(pauses.pauses) == set(Reason)
    for reason, expected_pause in expected_pauses.items():
        assert pauses.get_pause_for(reason) == expected_pause


def test_reason_pause_types_have_distinct_ids() -> None:
    assert ReasonPauseTypes.NORMAL.value.id == "normal"
    assert ReasonPauseTypes.SHORTER.value.id == "shorter"
    assert ReasonPauseTypes.NORMAL.value.label == "Normal"
    assert ReasonPauseTypes.NORMAL.value.menu_label == "normal"
    assert ReasonPauseTypes.NORMAL.value.description == ""
    assert ReasonPauseTypes.SHORTER.value.label == "Shorter"
    assert ReasonPauseTypes.SHORTER.value.menu_label == "shorter"
    assert ReasonPauseTypes.SHORTER.value.description == (
        'Enabling "Generate > Limit silence gaps" works well with this setting'
    )


def test_reason_pause_type_default_and_lookup() -> None:
    assert ReasonPauseTypes.default() is ReasonPauseTypes.NORMAL
    assert ReasonPauseTypes.get_by_id("normal") is ReasonPauseTypes.NORMAL
    assert ReasonPauseTypes.get_by_id("shorter") is ReasonPauseTypes.SHORTER
    assert ReasonPauseTypes.get_by_id("unknown") is None


def test_get_pause_for_requires_a_known_reason() -> None:
    pauses = ReasonPauseTypes.NORMAL.value

    with pytest.raises(KeyError):
        pauses.get_pause_for("sentence")  # type: ignore[arg-type]


def test_project_reason_pauses_defaults_for_missing_or_invalid_id() -> None:
    assert Project.model_validate({}).reason_pauses is ReasonPauseTypes.NORMAL.value
    assert (
        Project.model_validate({"reason_pauses": "unknown"}).reason_pauses
        is ReasonPauseTypes.NORMAL.value
    )


def test_project_reason_pauses_json_round_trip() -> None:
    project = Project.model_validate({"reason_pauses": "shorter"})

    assert project.reason_pauses is ReasonPauseTypes.SHORTER.value

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["reason_pauses"] == "shorter"

    reloaded = Project.model_validate(payload)
    assert reloaded.reason_pauses is ReasonPauseTypes.SHORTER.value
