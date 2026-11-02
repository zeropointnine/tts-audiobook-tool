import pytest

from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu_shared import (
    OutputUsage,
    StatusLine,
    estimate_tokens,
    format_token_count,
)
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestProgress, LlmRequestResult
from tts_audiobook_tool.text_util import strip_ansi_codes


@pytest.mark.parametrize("chars, expected", [(0, 0), (1, 1), (4, 1), (5, 2), (5_000, 1_250)])
def test_estimate_tokens(chars, expected):
    assert estimate_tokens(chars) == expected


@pytest.mark.parametrize("answer, reasoning, expected", [
    (5_000, 8_000, "Receiving: ~1.3K tokens"),
    (5_368, 0, "Receiving: ~1.3K tokens"),
    (0, 5_368, "Thinking: ~1.3K tokens"),
    (0, 3_200, "Thinking: ~0.8K tokens"),
    (1, 0, "Receiving: ~0.0K tokens"),
    (0, 0, "Waiting for response: 1s"),
])
def test_progress_displays_token_estimates(capsys, answer, reasoning, expected):
    # Both menus share this display; answer progress takes precedence over thinking.
    status = StatusLine()
    progress = LlmRequestProgress(answer_chars=answer, reasoning_chars=reasoning, elapsed=1)
    status.update_progress(progress)
    assert strip_ansi_codes(capsys.readouterr().out) == f"\r{expected}"
    status.update_progress(progress)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("tokens, expected", [
    (0, "0.0K"), (999, "1.0K"), (1_000, "1.0K"), (1_249, "1.2K"),
    (1_250, "1.3K"), (1_342, "1.3K"), (12_345, "12.3K"), (999_999, "1,000.0K"),
])
def test_format_token_count(tokens, expected):
    # One decimal retains progress detail; halfway values round up consistently.
    assert format_token_count(tokens, realtime=True) == expected


def test_mixed_reasoning_usage_retains_reported_and_estimated_counts():
    # Only missing components are estimated; reported reasoning isn't added to output twice.
    usage = OutputUsage()
    usage.add(LlmRequestResult(reply="a", reasoning="", elapsed=1,
                               output_tokens=100, reasoning_tokens=80))
    usage.add(LlmRequestResult(reply="a" * 8, reasoning="r" * 12, elapsed=1,
                               output_tokens=10))
    assert usage.format() == "0.1K output tokens (~0.1K reasoning)"
    usage.add(LlmRequestResult(reply="", reasoning="", elapsed=1, cancelled=True),
              answer_chars=8, reasoning_chars=12)
    assert usage.format() == "~0.1K output tokens (~0.1K reasoning)"
    assert usage.output_tokens == 115
    assert usage.reasoning_tokens == 86
