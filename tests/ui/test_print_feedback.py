import pytest

from tts_audiobook_tool import util


@pytest.mark.parametrize("options, expected", [
    ({}, 0.3),
    ({"long_pause": True}, 1.7),
    ({"skip_pause": True, "long_pause": True}, 0.0),
    ({"is_error": True, "skip_pause": True}, 1.7),
])
def test_feedback_uses_independent_pause_constants(monkeypatch, options, expected):
    # Long pauses are independently configurable; skip wins unless this is an error.
    sleeps = []
    monkeypatch.setattr(util, "PRINT_FEEDBACK_PAUSE", 0.3)
    monkeypatch.setattr(util, "PRINT_FEEDBACK_LONG_PAUSE", 1.7)
    monkeypatch.setattr(util.time, "sleep", sleeps.append)
    monkeypatch.setattr(util, "printt", lambda *args, **kwargs: None)

    util.print_feedback("Feedback", **options)

    assert sleeps == [expected]
