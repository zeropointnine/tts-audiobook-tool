from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call, patch

from tts_audiobook_tool import generate_util
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.state import State

from generate_files_test_support import StubValidationResult, generate_files_mock_stack


def test_batch_heading_can_omit_the_leading_divider(monkeypatch) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))

    GenerateUtil.print_batch_heading([0], show_divider=False)

    assert len(output) == 2
    assert "Processing line 1" in output[0]
    assert output[1] == ""
    assert not any("---" in line for line in output)


def test_clean_quick_generation_retries_then_omits_metrics_summary(monkeypatch) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        max_retries=2,
        phrase_groups=[phrase_group],
        sound_segments=sound_segments,
        generate_range_string="all",
        save=MagicMock(return_value=""),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
        ),
    )
    results = [StubValidationResult(True), StubValidationResult(True), StubValidationResult(False)]
    force_random_seed_values: list[bool] = []

    def generate_and_validate_batch(**kwargs: object) -> list[StubValidationResult]:
        force_random_seed_values.append(kwargs["force_random_seed"])  # type: ignore[arg-type]
        return [results.pop(0)]

    with (
        generate_files_mock_stack(generate_and_validate_batch),
        patch.object(GenerateUtil, "print_batch_heading") as print_batch_heading,
    ):
        did_interrupt = GenerateUtil.generate_files(state, {0}, batch_size=1, is_regen=True)

    assert not did_interrupt
    assert not any("Elapsed:" in text for text in output)
    assert not any("Lines saved:" in text for text in output)
    assert print_batch_heading.call_args_list == [
        call(indices=[0], voice_index=None, show_divider=False),
        call(indices=[0], voice_index=None, show_divider=True),
        call(indices=[0], voice_index=None, show_divider=True),
    ]
    assert force_random_seed_values == [True, True, True]
    assert results == []
    assert project.generate_range_string == "none"
    project.save.assert_called_once_with()


def test_failed_quick_generation_keeps_metrics_summary(monkeypatch) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        max_retries=0,
        phrase_groups=[phrase_group],
        sound_segments=sound_segments,
        generate_range_string="all",
        save=MagicMock(return_value=""),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(
                stt_variant=None,
                stt_config=None,
                save_debug_files=False,
            ),
        ),
    )

    with generate_files_mock_stack(lambda **_: [StubValidationResult(True)]):
        did_interrupt = GenerateUtil.generate_files(
            state,
            {0},
            batch_size=1,
            is_regen=True,
        )

    assert not did_interrupt
    assert any("Elapsed:" in text for text in output)
    assert any("Lines saved, but with excess word errors:" in text for text in output)