from itertools import product
from types import SimpleNamespace

import pytest

from tts_audiobook_tool import util


@pytest.fixture
def feedback_output(monkeypatch):
    lines, pauses = [], []
    monkeypatch.setattr(util, "printt", lambda message="": lines.append(message))
    monkeypatch.setattr(util, "time", SimpleNamespace(sleep=pauses.append))
    return lines, pauses


@pytest.mark.parametrize("no_preformat,skip_pause,long_pause", list(product([False, True], repeat=3)))
def test_error_feedback_overrides_formatting_and_pause_flags(
    feedback_output, no_preformat, skip_pause, long_pause,
):
    lines, pauses = feedback_output
    util.print_feedback(
        "Problem",
        no_preformat=no_preformat,
        skip_pause=skip_pause,
        long_pause=long_pause,
        is_error=True,
    )
    assert lines == [util.COL_DIM_ITALICS + util.COL_ERROR + "Problem", ""]
    assert pauses == [2 * util.PRINT_FEEDBACK_PAUSE]


def test_error_feedback_preserves_end_value_and_extra_line_options(feedback_output):
    lines, pauses = feedback_output
    util.print_feedback("Problem:", "details", is_error=True, extra_line=False)
    assert lines == [
        util.COL_DIM_ITALICS + util.COL_ERROR + "Problem: "
        + util.Ansi.RESET + util.COL_DEFAULT + "details",
    ]
    assert pauses == [2 * util.PRINT_FEEDBACK_PAUSE]


@pytest.mark.parametrize("no_preformat,skip_pause,long_pause", list(product([False, True], repeat=3)))
def test_regular_feedback_preserves_existing_options(
    feedback_output, no_preformat, skip_pause, long_pause,
):
    lines, pauses = feedback_output
    util.print_feedback(
        "Message",
        no_preformat=no_preformat,
        skip_pause=skip_pause,
        long_pause=long_pause,
        is_error=False,
    )
    prefix = "" if no_preformat else util.COL_DIM_ITALICS
    assert lines == [prefix + "Message", ""]
    expected_pause = 0.0 if skip_pause else util.PRINT_FEEDBACK_PAUSE * (2 if long_pause else 1)
    assert pauses == [expected_pause]
