from contextlib import nullcontext
import os
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.menus.speaker_assign import assign_menu
from tts_audiobook_tool.menus.speaker_assign.assign_menu import (
    AssignData,
    ROSTER_TEXT_NOTICE,
    make_subheading,
    play_outcome_sound,
)
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.attribution_runner import RunOutcome
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestProgress, LlmRequestResult, LlmRequestUtil
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter, RosterFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book
from tests.speaker_assign.test_voice_apply import ROSTER_IDS, make_fixture


@pytest.mark.parametrize("stale", [False, True])
def test_roster_notice_explains_reuse_without_implying_assignments_were_deleted(stale):
    # Roster provenance is advisory, separate from invalidation of text-bound assignments.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    hashes = ("old text",) if stale else UtteranceUtil.section_text_hashes(book)
    roster = RosterFile(Roster(()), section_text_hashes=hashes)
    saved = AttributionsFile()
    data = AssignData(roster, saved, AssignPlan.make(book, saved))
    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book)))
    heading = make_subheading(state, data)
    assert (ROSTER_TEXT_NOTICE in heading) == stale
    if stale:
        assert f"{COL_DIM}{ROSTER_TEXT_NOTICE}" in heading
        assert f"{COL_ERROR}{ROSTER_TEXT_NOTICE}" not in heading


@pytest.mark.parametrize("outcome, has_failures, expected", [
    (RunOutcome.FINISHED, False, "done"),
    (RunOutcome.FINISHED, True, None),
    (RunOutcome.FAILED, False, None),
    (RunOutcome.STOPPED, False, None),
    (RunOutcome.CANCELLED, False, None),
])
def test_outcome_sound(monkeypatch, outcome, has_failures, expected):
    # Only a fully successful run plays a sound (done); errors and user stops play nothing.
    played: list[str] = []
    monkeypatch.setattr(app_support, "play_done_sound", lambda: played.append("done"))
    monkeypatch.setattr(app_support, "play_fatal_gen_sound", lambda: played.append("error"))
    play_outcome_sound(outcome, has_failures)
    assert played == ([expected] if expected else [])


def test_clear_deletes_saved_assignments_and_drops_the_item(monkeypatch, tmp_path):
    # "Clear" is offered while any results are saved; once confirmed, it deletes
    # them and the menu falls back to a plain "Start". Step 3's voice mappings
    # stay: the roster's character ids they map are unchanged.
    book, attributions = make_fixture()
    roster_file = RosterFile(Roster(tuple(RosterCharacter(id=id, name=id) for id in sorted(ROSTER_IDS))))
    path = SpeakerAssignFileUtil.attributions_path(str(tmp_path))
    os.makedirs(os.path.dirname(path))
    voices_path = SpeakerAssignFileUtil.voices_path(str(tmp_path))
    for file_path in (path, voices_path):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("{}")

    def load(_) -> AssignData:
        saved = attributions if os.path.exists(path) else AttributionsFile()
        return AssignData(roster_file, saved, AssignPlan.make(book, saved))

    state = cast(State, SimpleNamespace(project=SimpleNamespace(book=book, dir_path=str(tmp_path))))
    monkeypatch.setattr(SpeakerAssignFileUtil, "has_roster", lambda _: True)
    monkeypatch.setattr(assign_menu, "load_assign_data", load)
    monkeypatch.setattr(assign_menu, "printt", lambda *_: None)
    monkeypatch.setattr(assign_menu.ask, "ask_confirm", lambda *_: True)
    feedback: list[str] = []
    monkeypatch.setattr(assign_menu, "print_feedback", lambda s, **_: feedback.append(s))
    monkeypatch.setattr(MenuUtil, "print_heading", lambda *_, **__: None)
    monkeypatch.setattr(MenuUtil, "make_breadcrumb_text", lambda *_: "")
    captured = []
    def capture_menu(*_, items, heading, breadcrumb, **__):
        assert heading == "Dialog attribution"
        assert breadcrumb == "Dialog attribution"
        captured.append(items)

    monkeypatch.setattr(MenuUtil, "menu", capture_menu)
    assert assign_menu.make_step_label(state).startswith("Step 2: Dialog attribution")

    assign_menu.assign_menu(state)
    items_maker = captured[0]
    clear = items_maker(state)[-1]
    assert clear.label == "Clear"

    clear.handler(state, clear)

    assert not os.path.exists(path)
    assert os.path.exists(voices_path)
    assert feedback == ["Dialog assignments cleared"]
    assert [item.label for item in items_maker(state)] == ["Start"]


@pytest.mark.parametrize("reasoning_tokens, expected", [
    (400, "3.0K input tokens, 1.0K output tokens (0.8K reasoning)"),
    (None, "3.0K input tokens, 1.0K output tokens (~0.0K reasoning)"),
])
def test_run_usage_totals_accumulate_reasoning(reasoning_tokens, expected):
    # Each request contributes once, even if it failed or was discarded.
    totals = assign_menu.RunUsageTotals()
    for cancelled in (False, True):
        totals.add(LlmRequestResult(
            reply="a" * 10, reasoning="r" * 20, elapsed=1,
            input_tokens=1_500, output_tokens=500, reasoning_tokens=reasoning_tokens,
            cancelled=cancelled,
        ), answer_chars=5, reasoning_chars=5)
    assert totals.format() == expected


def test_run_usage_totals_mark_partial_metadata_and_count_observed_output():
    # A stopped stream may not return its partial reply; observed characters still count.
    totals = assign_menu.RunUsageTotals()
    totals.add(LlmRequestResult(reply="a" * 10, reasoning="", elapsed=1,
                                input_tokens=1_000, output_tokens=100))
    totals.add(LlmRequestResult(reply="", reasoning="", elapsed=1, cancelled=True),
               answer_chars=5, reasoning_chars=20)
    assert totals.format() == (
        "1.0K input tokens (partial; 1 request without usage), "
        "~0.1K output tokens "
        "(~0.0K reasoning)"
    )


@pytest.mark.parametrize("input_tokens, output_tokens, expected", [
    (None, None, "input tokens unavailable, ~0.0K output tokens"),
    (0, 0, "0.0K input tokens, 0.0K output tokens"),
    (1_234, 50, "1.2K input tokens, 0.1K output tokens"),
])
def test_run_usage_totals_non_reasoning_and_missing_usage(input_tokens, output_tokens, expected):
    # Missing output is estimated; reported zero remains authoritative.
    totals = assign_menu.RunUsageTotals()
    totals.add(LlmRequestResult(reply="a" * 10, reasoning="", elapsed=1,
                                input_tokens=input_tokens, output_tokens=output_tokens))
    assert totals.format() == expected


def test_large_totals_round_all_usage_components_to_whole_k():
    # Input, output and reasoning follow the same non-realtime rounding rule.
    totals = assign_menu.RunUsageTotals()
    totals.add(LlmRequestResult(reply="", reasoning="", elapsed=1,
                                input_tokens=33_000, output_tokens=20_564, reasoning_tokens=17_303))
    assert totals.format() == "33K input tokens, 21K output tokens (17K reasoning)"


# Request formatting (request_kind) is independent of the run outcome; failures
# only change the FINISHED summary, so each axis value appears once or twice.
@pytest.mark.parametrize(("request_kind", "outcome", "has_failures"), [
    ("reported", RunOutcome.FINISHED, False),
    ("estimated", RunOutcome.FINISHED, True),
    ("cancelled", RunOutcome.STOPPED, False),
    ("error", RunOutcome.CANCELLED, True),
    ("reported", RunOutcome.FAILED, True),
])
def test_run_summary_prints_usage_and_starts_after_blank_line(
        monkeypatch, tmp_path, capsys, outcome, has_failures, request_kind):
    # Exercise the UI send wrapper: retries/follow-ups count, and every exit prints totals.
    book = make_book([('"Hello."', Reason.PHRASE_QUOTE_END, DIALOG)])
    saved = AttributionsFile()
    data = AssignData(RosterFile(Roster(())), saved, AssignPlan.make(book, saved))
    state = cast(State, SimpleNamespace(prefs=None, project=SimpleNamespace(
        book=book, dir_path=str(tmp_path),
    )))
    monkeypatch.setattr(assign_menu, "load_assign_data", lambda _: data)
    monkeypatch.setattr(assign_menu.RosterUtil, "is_stale", lambda *_: False)
    monkeypatch.setattr(assign_menu, "print_llm_info", lambda _: None)
    monkeypatch.setattr(assign_menu, "play_outcome_sound", lambda *_: None)
    monkeypatch.setattr(assign_menu, "prevent_system_sleep", nullcontext)
    monkeypatch.setattr(assign_menu.ask, "ask_enter_to_continue", lambda *_: None)
    monkeypatch.setattr(assign_menu, "Interrupts", lambda: SimpleNamespace(
        set=lambda _: None, clear=lambda: None, press_count=0,
    ))
    calls = []

    def send_cancellable(llm, message, **kwargs):
        calls.append(message)
        kwargs["on_poll"](LlmRequestProgress(answer_chars=400, reasoning_chars=40_000, elapsed=1))
        return LlmRequestResult(
            reply="a" * 400, reasoning="r" * 40_000, elapsed=85,
            input_tokens=33_000 if request_kind != "estimated" else None,
            output_tokens=20_564 if request_kind != "estimated" else None,
            reasoning_tokens=17_303 if request_kind != "estimated" else None,
            cancelled=request_kind == "cancelled",
            error=RuntimeError("failed") if request_kind == "error" else None,
        )

    class Runner:
        def __init__(self, **kwargs):
            self.send = kwargs["send"]
            self.listener = kwargs["listener"]

        def run(self):
            self.send(None, "first")
            self.send(None, "follow-up or retry")
            self.listener.chunk_saved(None, None, 85)
            # Simulate an unterminated terminal line immediately before run returns.
            print("last chunk", end="")
            return SimpleNamespace(outcome=outcome, saved_count=1, failures=["failed"] if has_failures else [],
                                   abandoned=False, error="test error", attributions=saved)

    monkeypatch.setattr(LlmRequestUtil, "send_cancellable", send_cancellable)
    monkeypatch.setattr(assign_menu, "AttributionRunner", Runner)

    assign_menu.run_assign(state)

    output = strip_ansi_codes(capsys.readouterr().out)
    label = {RunOutcome.FINISHED: "Done", RunOutcome.FAILED: "Failed"}.get(outcome, "Stopped")
    assert f"last chunk\n\n{label}:" in output
    if request_kind == "estimated":
        assert "Totals: input tokens unavailable, ~20K output tokens (~20K reasoning)\n" in output
        done = "Done: input tokens unavailable, ~10K output tokens (~10.0K reasoning), 1m25s\n"
    else:
        assert "Totals: 66K input tokens, 41K output tokens (35K reasoning)\n" in output
        done = "Done: 33K input tokens, 21K output tokens (17K reasoning), 1m25s\n"
    if request_kind in ("reported", "estimated"):
        assert output.count(done) == 2
        # Each request summary replaces (rather than follows) its live progress line.
        assert output.count("\r" + done) == 2
    else:
        assert done not in output
    assert "Done: 1m25s" not in output
    assert "select Continue" not in output
    assert calls == ["first", "follow-up or retry"]


@pytest.mark.parametrize("count, expected", [
    (0, "0.0K"), (499, "0.5K"), (500, "0.5K"), (1_499, "1.5K"),
    (1_500, "1.5K"), (10_000, "10.0K"), (10_001, "10K"), (12_345, "12K"), (12_500, "13K"), (999_500, "1,000K"),
])
def test_summary_token_counts_use_whole_k_only_over_10k(count, expected):
    # Halfway values round up rather than using Python's nearest-even rounding.
    assert assign_menu.format_token_count(count) == expected
