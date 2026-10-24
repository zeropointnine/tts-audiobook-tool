"""Step 2: whole-book speaker attribution (see plans/llm-speaker-attribution.md)."""
from collections import Counter
from dataclasses import dataclass
import time

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_support.system_sleep import prevent_system_sleep
from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu_shared import (
    CHARS_PER_TOKEN,
    LLM_DATA_NOTICE,
    StatusLine,
    check_llm_configured,
    format_elapsed,
    format_token_count,
    print_llm_info,
)
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan, PlannedChunk
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution
from tts_audiobook_tool.speaker_assign.attribution_prompt_util import (
    ATTRIBUTION_SYSTEM_PROMPT,
    AttributionPromptUtil,
)
from tts_audiobook_tool.speaker_assign.attribution_runner import (
    AttributionRunListener,
    AttributionRunner,
    RunOutcome,
)
from tts_audiobook_tool.speaker_assign.attribution_summary import AttributionSummary
from tts_audiobook_tool.speaker_assign.llm_request_util import (
    CancelEvent,
    LlmRequestProgress,
    LlmRequestResult,
    LlmRequestUtil,
)
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterFile
from tts_audiobook_tool.speaker_assign.roster_util import RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, OTHER, UNKNOWN
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import make_noun, print_feedback, printt


# Observed output per dialog passage, reasoning included (deepseek-flash, thinking
# enabled); only a rough guide for other models
OUTPUT_TOKENS_PER_UTTERANCE = 66
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
    label = "Step 2: Assign dialog to characters"
    if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
        return label + f" {COL_DIM}(requires step 1)"
    data = load_assign_data(state)
    if isinstance(data, str) or not data.plan.chunks:
        return label
    if data.plan.is_complete:
        return label + f" {COL_DIM}(completed)"
    if data.plan.done_count:
        return label + f" {COL_DIM}({data.plan.done_count}/{len(data.plan.chunks)} chunks)"
    return label


def assign_menu(state: State) -> None:

    if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
        print_feedback("Requires step 1 (create character roster)", is_error=True)
        return
    data = load_assign_data(state)
    if isinstance(data, str):
        ask.ask_error(data)
        return

    # The subheading's progress and estimate walk the whole book: computed on
    # entry and after each run, not on every redraw
    current: AssignData = data
    subheading = make_subheading(state, current)

    def refresh() -> None:
        nonlocal current, subheading
        fresh = load_assign_data(state)
        if not isinstance(fresh, str):
            current, subheading = fresh, make_subheading(state, fresh)

    def on_start(_: State, __: MenuItem) -> None:
        if not check_llm_configured(state):
            return
        MenuUtil.print_screen_heading(state, "Assigning dialog to characters", breadcrumb="Assigning dialog to characters")
        run_assign(state)
        refresh()

    def on_view(_: State, __: MenuItem) -> None:
        MenuUtil.print_screen_heading(state, "Dialog assignments", breadcrumb="Dialog assignments")
        if print_assign_results(state):
            ask.ask_enter_to_continue()

    def items_maker(_: State) -> list[MenuItem]:
        plan = current.plan
        items = []
        if not plan.is_complete:
            items.append(MenuItem("Continue" if plan.done_count else "Start", on_start))
        if plan.done_count:
            items.append(MenuItem("View results", on_view))
        return items

    MenuUtil.menu(
        state,
        heading="Assign dialog to characters",
        items=items_maker,
        subheading=lambda _: subheading,
        breadcrumb="Assign dialog to characters",
    )


def make_subheading(state: State, data: AssignData) -> str:
    book = state.project.book
    plan = data.plan
    lines = [ASSIGN_DESC, ""]
    if not plan.chunks:
        lines.append(f"{COL_ERROR}No dialog found in the project text")
        return "\n".join(lines)
    if plan.is_complete:
        lines.append(f"Progress: {COL_ACCENT}all {len(plan.chunks)} chunks done{COL_DEFAULT}")
    else:
        remaining = plan.remaining
        utterances = sum(chunk.utterance_count for chunk in remaining)
        input_tokens, output_tokens = estimate_tokens(book, data.roster_file.roster, data.attributions, remaining)
        if plan.done_count:
            lines.append(f"Progress: {COL_ACCENT}{plan.done_count} of {len(plan.chunks)}{COL_DEFAULT} chunks done")
        lines.append(
            f"To do: {COL_ACCENT}{len(remaining)}{COL_DEFAULT} {make_noun('chunk', 'chunks', len(remaining))}, "
            f"{utterances:,} dialog passages")
        lines.append(
            f"Estimated input size: {COL_ACCENT}~{format_token_count(input_tokens)} tokens{COL_DEFAULT}; "
            f"output: ~{format_token_count(output_tokens)} tokens {COL_DIM}(including reasoning; varies by model)")
    if RosterUtil.is_stale(data.roster_file.section_text_hashes, book):
        lines.append(f"{COL_DIM}{ROSTER_TEXT_NOTICE}")
    return "\n".join(lines)


def estimate_tokens(
        book: Book, roster: Roster, attributions: AttributionsFile, chunks: list[PlannedChunk],
) -> tuple[int, int]:
    """Approximate input and output tokens for attributing `chunks`; each
    request repeats the instructions, roster and some preceding context."""
    speakers_by_line = attributions.speakers_by_line(book)
    chars = sum(
        len(ATTRIBUTION_SYSTEM_PROMPT)
        + len(AttributionPromptUtil.build_message(book, chunk.start, chunk.end, roster, speakers_by_line))
        for chunk in chunks
    )
    output = sum(chunk.utterance_count for chunk in chunks) * OUTPUT_TOKENS_PER_UTTERANCE
    return chars // CHARS_PER_TOKEN, output


def run_assign(state: State) -> None:
    """
    Attributes the remaining chunks (AttributionRunner), saving after each.
    Control-c once stops after the current chunk; twice stops immediately
    (the current chunk is not saved).
    """
    book = state.project.book
    prefs = state.prefs
    data = load_assign_data(state)
    if isinstance(data, str):
        ask.ask_error(data)
        return
    plan = data.plan
    if plan.is_complete:
        ask.ask_enter_to_continue("All dialog has already been assigned.")
        return

    remaining = plan.remaining
    title = book.title.strip() or "(untitled)"
    printt(
        f"{title}: {len(plan.chunks)} chunks; {plan.done_count} done, {len(remaining)} to do "
        f"({sum(chunk.utterance_count for chunk in remaining):,} dialog passages)")
    if RosterUtil.is_stale(data.roster_file.section_text_hashes, book):
        printt(f"{COL_DIM}{ROSTER_TEXT_NOTICE}")
    print_llm_info(prefs)
    printt(f"{COL_DIM}Press control-c to stop after the current chunk (twice to stop immediately)")
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
        attributions=data.attributions,
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
    saved = f"{result.saved_count} {make_noun('chunk', 'chunks', result.saved_count)} saved this run"
    match result.outcome:
        case RunOutcome.FINISHED:
            if result.failures:
                count = len(result.failures)
                printt(f"{saved}. {COL_ERROR}{count} {make_noun('chunk', 'chunks', count)} failed; "
                       f"select Continue to retry.")
            else:
                printt(f"{saved}. {COL_ACCENT}All chunks done.")
        case RunOutcome.STOPPED:
            printt(f"Stopped: {saved}. Select Continue to resume.")
        case RunOutcome.CANCELLED:
            printt(f"Stopped: {saved}; the current chunk was not saved. Select Continue to resume.")
            if result.abandoned:
                printt(f"{COL_DIM}Its request was abandoned before the response started; "
                       f"the LLM provider may still process it.")
        case RunOutcome.FAILED:
            printt(f"{saved}.")
            printt(f"{COL_ERROR}{result.error}")
    printt(f"{COL_DIM}Elapsed: {format_elapsed(time.monotonic() - started)}")
    printt()

    if result.attributions.ranges:
        print_assign_results(state)
    ask.ask_enter_to_continue()


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
        label = UtteranceUtil.range_label(self.book, chunk.start, chunk.end)
        printt(f"[{chunk.number}/{len(plan.chunks)}] {label}: "
               f"{chunk.utterance_count:,} dialog passages")

    def note(self, text: str) -> None:
        self.status.end()
        printt(f"  {COL_DIM}{text}")

    def retrying(self, attempt: int, delay: float, error: str) -> None:
        self.status.end()
        printt(f"  {COL_ERROR}Attempt {attempt} failed:{COL_DEFAULT} {error}")
        printt(f"  {COL_DIM}Retrying in {delay:.0f}s")

    def chunk_saved(self, chunk: PlannedChunk, attribution: RangeAttribution, elapsed: float) -> None:
        counts = AttributionSummary.counts_text(attribution_counts(attribution))
        printt(f"  {COL_DIM}Done in {format_elapsed(elapsed)}: {counts}")

    def chunk_failed(self, chunk: PlannedChunk, error: str) -> None:
        self.status.end()
        printt(f"  {COL_ERROR}Failed:{COL_DEFAULT} {error}")


def attribution_counts(attribution: RangeAttribution) -> Counter[str]:
    return Counter(SpeakerLabelUtil.counting_id(u.speaker) for u in attribution.utterances)


def print_assign_results(state: State) -> bool:
    """
    Prints the saved speaker ids: per-speaker counts (roster characters by
    count, then the reserved answers) and recurring speakers not in the
    roster. Shows an error and returns False if the saved
    data can't be loaded.
    """
    data = load_assign_data(state)
    if isinstance(data, str):
        ask.ask_error(data)
        return False
    book = state.project.book
    roster = data.roster_file.roster
    plan = data.plan
    summary = AttributionSummary.make(data.attributions, book)

    total = summary.attributed + summary.unattributed
    printt(
        f"Dialog assigned: {COL_ACCENT}{summary.attributed:,}{COL_DEFAULT} of {total:,} dialog passages "
        f"{COL_DIM}({plan.done_count} of {len(plan.chunks)} chunks)")
    if not summary.attributed:
        return True

    names = {character.id: character.name for character in roster.characters}
    characters = sorted(
        ((label, count) for label, count in summary.counts.items() if label in names),
        key=lambda item: (-item[1], item[0]))
    reserved = [(label, summary.counts[label]) for label in (OTHER, UNKNOWN, NOT_SPEECH) if summary.counts[label]]
    # Labels neither in the roster nor reserved shouldn't occur (a new roster
    # discards saved ids), but are listed rather than hidden
    others = sorted((label, count) for label, count in summary.counts.items()
                    if label not in names and label not in RESERVED_LABEL_DESCRIPTIONS)

    rows = [*characters, *others, *reserved]
    id_width = max(len("id"), *(len(label) for label, _ in rows))
    printt()
    printt(f"{COL_DIM}{'count':>6}  {'id':<{id_width}}  name")
    for label, count in rows:
        name = names.get(label) or f"{COL_DIM}{RESERVED_LABEL_DESCRIPTIONS.get(label, '')}"
        printt(f"{count:>6,}  {COL_ACCENT}{label:<{id_width}}{COL_DEFAULT}  {name}")
    printt()

    if summary.recurring_others:
        printt(f"Frequent speakers not in the roster: {AttributionSummary.counts_text(summary.recurring_others)}")
        printt(f"{COL_DIM}(counted under {OTHER}; re-creating the roster may pick them up)")
    return True


RESERVED_LABEL_DESCRIPTIONS = {
    OTHER: "characters not in the roster",
    UNKNOWN: "speaker couldn't be determined",
    NOT_SPEECH: "quoted text that isn't speech",
}


ASSIGN_DESC = \
f"""Sends the book to the LLM in chunks of about 5,000 words, one request per chunk,
in reading order. The LLM assigns each line of dialog to a character from the
roster. Results are saved after every chunk, so a run can be stopped and continued
later. The project text is not changed.

Depending on the model, a whole book may take several minutes.
{LLM_DATA_NOTICE}
{COL_DIM}To redo all dialog assignments, re-create the roster (step 1).{COL_DEFAULT}"""
