"""Step 2: whole-book speaker attribution (see plans/llm-speaker-attribution.md)."""
from dataclasses import dataclass
import time

from tts_audiobook_tool import app_support, ask
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_support.system_sleep import prevent_system_sleep
from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign.characters_view import print_characters_info
from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu_shared import (
    StatusLine,
    check_llm_configured,
    print_llm_info,
)
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan, PlannedChunk
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution
from tts_audiobook_tool.speaker_assign.attribution_runner import (
    AttributionRunListener,
    AttributionRunner,
    RunOutcome,
)
from tts_audiobook_tool.speaker_assign.llm_request_util import (
    CancelEvent,
    LlmRequestProgress,
    LlmRequestResult,
    LlmRequestUtil,
)
from tts_audiobook_tool.speaker_assign.roster import RosterFile
from tts_audiobook_tool.speaker_assign.roster_util import RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import (
    duration_string,
    make_hotkey_string,
    make_noun,
    print_accent_separator,
    print_feedback,
    printt,
)


ROSTER_TEXT_NOTICE = (
    "The character roster was created from an earlier version of the text. "
    "You can keep using it, or recreate it in Step 1."
)


@dataclass(frozen=True)
class AssignData:
    """The saved roster and attributions, and the resulting plan."""
    roster_file: RosterFile
    attributions: AttributionsFile
    plan: AssignPlan


def load_assign_data(state: State) -> AssignData | str:
    """The saved step 1 and 2 results, or an error string."""
    project_dir = state.project.dir_path
    roster_file = SpeakerAssignFileUtil.load_roster_file(project_dir)
    if isinstance(roster_file, str):
        return roster_file
    attributions = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(attributions, str):
        return attributions
    return AssignData(roster_file, attributions, AssignPlan.make(state.project.book, attributions))


def make_step_label(state: State) -> str:
    """The parent menu's label for step 2, with its progress."""
    label = "Step 2: Run dialog attribution"
    if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
        return label + f" {COL_DIM}(requires step 1)"
    data = load_assign_data(state)
    if isinstance(data, str) or not data.plan.chunks:
        return label
    if data.plan.is_complete:
        return label + f" {COL_DIM}(completed)"
    if data.plan.done_count:
        return label + f" {COL_DIM}({data.plan.done_count} of {len(data.plan.chunks)} chunks complete)"
    return label


def assign_menu(state: State) -> None:

    if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
        print_feedback("Requires step 1 (generate character roster)", is_error=True)
        return
    data = load_assign_data(state)
    if isinstance(data, str):
        ask.ask_error(data)
        return

    # The plan and stale check behind the subheading walk the whole book: computed on
    # entry and after each run, not on every redraw
    current: AssignData = data
    subheading = make_subheading(state, current)

    def refresh() -> None:
        nonlocal current, subheading
        fresh = load_assign_data(state)
        if not isinstance(fresh, str):
            current, subheading = fresh, make_subheading(state, fresh)

    # No new heading for the handlers below: confirmations and runs print
    # directly below the menu

    def on_start(_: State, __: MenuItem) -> None:
        if not check_llm_configured(state):
            return
        run_assign(state)
        refresh()

    def on_start_over(_: State, __: MenuItem) -> None:
        if not check_llm_configured(state):
            return
        printt("Existing dialog assignments will be replaced.")
        printt()
        if not ask.ask_confirm(""):
            return
        run_assign(state, start_over=True)
        refresh()

    def on_clear(_: State, __: MenuItem) -> None:
        printt("Existing dialog assignments will be deleted.")
        printt()
        if not ask.ask_confirm(""):
            return
        error = SpeakerAssignFileUtil.delete_attributions_file(state.project.dir_path)
        refresh()
        if error:
            ask.ask_error(error)
            return
        print_feedback("Dialog assignments cleared")

    def items_maker(_: State) -> list[MenuItem]:
        plan = current.plan
        if not plan.chunks:
            items = []
        elif plan.is_complete:
            items = [MenuItem("Start over", on_start_over)]
        elif plan.done_count:
            items = [MenuItem("Continue", on_start), MenuItem("Start over", on_start_over)]
        else:
            items = [MenuItem("Start", on_start)]
        # Any saved results, including ones the text has since invalidated
        if current.attributions.ranges:
            items.append(MenuItem("Clear", on_clear))
        return items

    MenuUtil.menu(
        state,
        heading="Run dialog attribution",
        items=items_maker,
        subheading=lambda _: subheading,
        breadcrumb="Run dialog attribution",
    )


def make_subheading(state: State, data: AssignData) -> str:
    book = state.project.book
    plan = data.plan
    lines = [ASSIGN_DESC, ""]
    if not plan.chunks:
        lines.append(f"{COL_ERROR}No dialog found in the project text")
        return "\n".join(lines)
    if not plan.is_complete:
        lines.append(f"{COL_DIM}{len(plan.remaining)} of {len(plan.chunks)} chunks remaining")
    if RosterUtil.is_stale(data.roster_file.section_text_hashes, book):
        lines.append(f"{COL_DIM}{ROSTER_TEXT_NOTICE}")
    return "\n".join(lines)


def run_assign(state: State, start_over: bool = False) -> None:
    """
    Attributes the remaining chunks (AttributionRunner), saving after each.
    Control-c once stops after the current chunk; twice stops immediately
    (the current chunk is not saved).

    With start_over, every chunk is attributed again, starting from no
    results. The saved results stay on disk until the first new chunk is
    saved over them, so a run that fails or is cancelled before then keeps
    them.
    """
    book = state.project.book
    prefs = state.prefs
    data = load_assign_data(state)
    if isinstance(data, str):
        ask.ask_error(data)
        return
    attributions = AttributionsFile() if start_over else data.attributions
    if AssignPlan.make(book, attributions).is_complete:
        ask.ask_enter_to_continue("All dialog has already been assigned.")
        return

    if RosterUtil.is_stale(data.roster_file.section_text_hashes, book):
        printt(f"{COL_DIM}{ROSTER_TEXT_NOTICE}")
    print_llm_info(prefs)
    printt()
    printt(f"Press {make_hotkey_string('CTRL-C')} to interrupt")
    printt()

    status = StatusLine()
    stop = StopWatcher(status)

    def send(llm: LlmSession, message: str) -> LlmRequestResult:
        def on_poll(progress: LlmRequestProgress) -> None:
            stop.presses()
            status.update_progress(progress)

        def on_cancel_event(_: CancelEvent) -> None:
            status.end()
            printt(f"{COL_ERROR}Control-C pressed again, stopping now")

        result = LlmRequestUtil.send_cancellable(
            llm, message,
            # The first press only stops the run after the current chunk
            cancel_requests=lambda: max(0, stop.presses() - 1),
            on_poll=on_poll,
            on_cancel_event=on_cancel_event,
            defer_until_streaming=False,
        )
        # Per-chunk results are printed instead
        status.clear()
        return result

    runner = AttributionRunner(
        book=book,
        roster=data.roster_file.roster,
        attributions=attributions,
        project_dir=state.project.dir_path,
        prefs=prefs,
        send=send,
        stop_requested=lambda: stop.presses() >= 1,
        listener=PrintingListener(book, status),
    )
    started = time.monotonic()
    Interrupts().set("llm")
    try:
        with prevent_system_sleep():
            result = runner.run()
    finally:
        Interrupts().clear()
    status.end()

    printt()
    elapsed = f"elapsed: {duration_string(time.monotonic() - started)}"
    # Starting over replaces the saved results only once a chunk is saved;
    # until then the previous results stand and there's nothing to continue
    kept_previous = start_over and result.saved_count == 0
    retry_hint = "" if kept_previous else " (select Continue to retry)"
    resume_hint = "" if kept_previous else "select Continue to resume, "
    match result.outcome:
        case RunOutcome.FINISHED:
            if result.failures:
                count = len(result.failures)
                printt(f"{COL_DEFAULT}Done: {COL_ERROR}{count} {make_noun('chunk', 'chunks', count)} failed"
                       f"{COL_DIM}{retry_hint}, {elapsed}")
            else:
                printt(f"{COL_DEFAULT}Done: {COL_DIM}all chunks complete, {elapsed}")
        case RunOutcome.STOPPED:
            printt(f"{COL_DEFAULT}Stopped: {COL_DIM}{resume_hint}{elapsed}")
        case RunOutcome.CANCELLED:
            printt(f"{COL_DEFAULT}Stopped: {COL_DIM}the current chunk was not saved, {resume_hint}{elapsed}")
            if result.abandoned:
                printt(f"{COL_DIM}Its request was abandoned before the response started; "
                       f"the LLM provider may still process it.")
        case RunOutcome.FAILED:
            printt(f"{COL_DEFAULT}Failed: {COL_DIM}{elapsed}")
            printt(f"{COL_ERROR}{result.error}")
    if kept_previous:
        printt(f"{COL_DIM}No chunks were saved, so the previous dialog assignments were kept.")
    printt()
    play_outcome_sound(result.outcome, bool(result.failures))

    if result.attributions.ranges:
        print_accent_separator("Character/dialog results")
        print_characters_info(state)
    ask.ask_enter_to_continue()


def play_outcome_sound(outcome: RunOutcome, has_failures: bool) -> None:
    """
    The done sound for a run that completed every chunk. No sound for one that
    ended in errors (aborted, or finished with failed chunks) or that the user stopped.
    """
    if outcome == RunOutcome.FINISHED and not has_failures:
        app_support.play_done_sound()


class StopWatcher:
    """Control-c presses since the run started; reports the first one."""

    def __init__(self, status: StatusLine) -> None:
        self.status = status
        self.reported = False

    def presses(self) -> int:
        count = Interrupts().press_count
        if count and not self.reported:
            self.reported = True
            self.status.end()
            printt(f"{COL_ERROR}Control-C pressed: will stop after the current chunk (press again to stop now)")
        return count


class PrintingListener(AttributionRunListener):

    def __init__(self, book: Book, status: StatusLine) -> None:
        self.book = book
        self.status = status

    def chunk_started(self, chunk: PlannedChunk, plan: AssignPlan) -> None:
        self.status.end()
        # User-facing line numbers are 1-based and inclusive
        printt(f"{COL_DEFAULT}Chunk {chunk.number} of {len(plan.chunks)}: "
               f"{COL_DIM}Lines {chunk.start + 1}-{chunk.end}, {chunk.utterance_count:,} dialog passages")

    def note(self, text: str) -> None:
        self.status.end()
        printt(f"  {COL_DIM}{text}")

    def retrying(self, attempt: int, delay: float, error: str) -> None:
        self.status.end()
        printt(f"  {COL_ERROR}Attempt {attempt} failed:{COL_DEFAULT} {error}")
        printt(f"  {COL_DIM}Retrying in {delay:.0f}s")

    def chunk_saved(self, chunk: PlannedChunk, attribution: RangeAttribution, elapsed: float) -> None:
        printt(f"{COL_DEFAULT}Done: {COL_DIM}{duration_string(elapsed)}")

    def chunk_failed(self, chunk: PlannedChunk, error: str) -> None:
        self.status.end()
        printt(f"  {COL_ERROR}Failed:{COL_DEFAULT} {error}")


ASSIGN_DESC = \
f"""The LLM assigns each line of dialog to a character from the roster. Results
are saved after each chunk, so a run can be stopped and resumed later."""
