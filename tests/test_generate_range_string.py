from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil


@pytest.mark.parametrize("value", ["", "all", "a", " ALL "])
def test_parse_all_generation_range_values(value: str) -> None:
    indices, warnings = RangeStringUtil.parse_ranges_string(value, 3)

    assert indices == {0, 1, 2}
    assert warnings == []


@pytest.mark.parametrize("value", ["none", "None", " NONE "])
def test_parse_none_generation_range_values(value: str) -> None:
    indices, warnings = RangeStringUtil.parse_ranges_string(value, 3)

    assert indices == set()
    assert warnings == []


def test_make_empty_generation_range_uses_canonical_none() -> None:
    assert RangeStringUtil.make_ranges_string(set(), 3) == "none"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", {0, 1, 2}),
        ("all", {0, 1, 2}),
        ("none", set()),
        ("2-3", {1, 2}),
    ],
)
def test_project_generation_range_interpretation(
    value: str, expected: set[int]
) -> None:
    project = cast(
        Project,
        SimpleNamespace(generate_range_string=value, phrase_groups=[1, 2, 3]),
    )

    assert ProjectUtil.get_indices_to_generate(project) == expected


@pytest.mark.parametrize("value", ["none", "None", " NONE "])
def test_loaded_none_generation_range_is_canonicalized(value: str) -> None:
    project = Project.model_validate({"generate_range": value})

    assert project.generate_range_string == "none"


@pytest.mark.parametrize("value", ["", "all", "a", " ALL "])
def test_loaded_all_generation_range_uses_legacy_empty_string(value: str) -> None:
    project = Project.model_validate({"generate_range": value})

    assert project.generate_range_string == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("9-99", set(range(8, 50))),
        ("-99", set(range(0, 50))),
    ],
)
def test_parse_ranges_string_clamps_range_end_with_warning(value: str, expected: set[int]) -> None:
    indices, warnings = RangeStringUtil.parse_ranges_string(value, 50)

    assert indices == expected
    assert warnings == ["Clamped range end to 50"]


def test_parse_ranges_string_no_warning_when_range_within_bounds() -> None:
    indices, warnings = RangeStringUtil.parse_ranges_string("5-10", 50)

    assert indices == set(range(4, 10))
    assert warnings == []


def test_parse_ranges_string_mixed_tokens_clamp_and_out_of_range() -> None:
    indices, warnings = RangeStringUtil.parse_ranges_string("2, 9-99", 50)

    assert indices == {1, *range(8, 50)}
    assert warnings == ["Clamped range end to 50"]
