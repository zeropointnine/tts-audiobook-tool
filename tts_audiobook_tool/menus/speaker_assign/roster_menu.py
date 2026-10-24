"""Step 1: the full-book roster request (see plans/llm-speaker-attribution.md)."""
import datetime
import math
import os

from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_DIM_ITALICS, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu_shared import (
    CHARS_PER_TOKEN,
    StatusLine,
    check_llm_configured,
    format_token_count,
    print_llm_info,
    save_temp_text,
)
from tts_audiobook_tool.speaker_assign.attribution_reply_util import AttributionReplyUtil
from tts_audiobook_tool.speaker_assign.llm_request_util import CancelEvent, LlmRequestUtil
from tts_audiobook_tool.speaker_assign.roster import RosterFile, RosterNarration
from tts_audiobook_tool.speaker_assign.roster_util import ROSTER_SYSTEM_PROMPT, RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import (
    duration_string,
    make_error_string,
    make_hotkey_string,
    print_accent_separator,
    printt,
)


def roster_menu(state: State) -> None:

    # Computed once on entry (walks the whole book), not on every redraw
    tokens = estimate_roster_input_tokens(state.project.book)
    # Always in thousands, rounded up, as it's a requirement
    subheading = ROSTER_DESC.replace(
        "%1", f"{COL_ACCENT}{math.ceil(tokens / 1000):,}k{COL_DEFAULT}"
    )

    def items_maker(_: State) -> list[MenuItem]:
        items = [MenuItem("Start", on_select_roster_start)]
        if SpeakerAssignFileUtil.has_roster(state.project.dir_path):
            items.append(MenuItem("View roster data" + make_roster_stale_marker(state), on_select_roster_view))
        return items

    MenuUtil.menu(
        state,
        heading="Create character roster",
        items=items_maker,
        subheading=subheading,
        breadcrumb="Create character roster",
    )


def make_roster_stale_marker(state: State) -> str:
    """Mark known stale roster data; unreadable data has no known provenance."""
    roster_file = SpeakerAssignFileUtil.load_roster_file(state.project.dir_path)
    if not isinstance(roster_file, str) and RosterUtil.is_stale(roster_file.section_text_hashes, state.project.book):
        return f"{COL_ERROR}*"
    return ""


def on_select_roster_start(state: State, _item: MenuItem) -> None:
    if not check_llm_configured(state):
        return
    # Same heading and breadcrumbs as the roster menu: the run screen takes its
    # place rather than nesting under it
    MenuUtil.print_heading(state, "Create character roster", breadcrumb_text=MenuUtil.make_breadcrumb_text(state))
    create_roster(state)


def on_select_roster_view(state: State, _item: MenuItem) -> None:
    MenuUtil.print_screen_heading(state, "Roster data", breadcrumb="Roster data")
    if print_roster_info(state):
        ask.ask_enter_to_continue()


# ---

def create_roster(state: State) -> None:
    """
    The full-book roster request: sends the book to the LLM (control-c
    cancels), then saves the roster and prints it. Nothing is saved or
    deleted unless a valid roster comes back.
    """
    project_dir = state.project.dir_path
    book = state.project.book
    prefs = state.prefs

    utterance_count = len(UtteranceUtil.find_utterances(book, 0, len(book.flat_phrase_groups)))
    if utterance_count == 0:
        message = "No dialog text found"
        if not book.segmentation_settings.dialog_segmentation:
            message += (
                "\nDialog segmentation must be enabled when importing text."
            )
        ask.ask_error(message)
        return

    # Replacing an existing roster invalidates speaker ids assigned with it
    has_roster = SpeakerAssignFileUtil.has_roster(project_dir)
    has_attributions = os.path.isfile(SpeakerAssignFileUtil.attributions_path(project_dir))
    if has_roster or has_attributions:
        if has_roster:
            printt("A roster already exists and will be replaced.")
        if has_attributions:
            printt(f"Dialog assignments from step 2 will be {COL_ERROR}discarded{COL_DEFAULT}.")
        printt()
        if not ask.ask_confirm(""):
            return

    try:
        llm = LlmRequestUtil.make_session(prefs, ROSTER_SYSTEM_PROMPT)
    except ValueError as e:
        ask.ask_error(f"Can't use the LLM settings: {e}")
        return

    message = RosterUtil.build_message(book)
    input_chars = len(ROSTER_SYSTEM_PROMPT) + len(message)

    printt(f"{COL_DEFAULT}Book text: {COL_DIM}{len(book.sections)} sections, {utterance_count:,} dialog passages")
    print_llm_info(prefs)
    printt(f"{COL_DEFAULT}Input size: {COL_DIM}{input_chars:,} chars "
           f"(~{format_token_count(input_chars // CHARS_PER_TOKEN)} tokens)")
    printt()
    printt(f"Press {make_hotkey_string('CTRL-C')} to cancel")
    printt()

    status = StatusLine()

    def on_cancel_event(event: CancelEvent) -> None:
        status.end()
        printt(f"{COL_ERROR}{CANCEL_EVENT_FEEDBACK[event]}")

    Interrupts().set("llm")
    try:
        result = LlmRequestUtil.send_cancellable(
            llm, message,
            cancel_requests=lambda: Interrupts().press_count,
            on_poll=status.update_progress,
            on_cancel_event=on_cancel_event,
        )
    finally:
        Interrupts().clear()
    if result.cancelled or result.error is not None:
        status.end()
    else:
        # The "Done" line below takes the progress line's place
        status.clear()

    if result.cancelled:
        feedback = "Cancelled. Nothing was saved."
        if result.abandoned:
            feedback += (f"\n{COL_DIM}The request was abandoned before the response started; "
                         f"the LLM provider may still process it.")
        ask.ask_enter_to_continue(feedback)
        return
    if result.error is not None:
        ask.ask_error(LlmRequestUtil.describe_error(result.error, prefs))
        return

    printt(f"{COL_DEFAULT}Done: {COL_DIM}{len(result.reply):,} answer chars, "
           f"{len(result.reasoning):,} reasoning chars, elapsed: {duration_string(result.elapsed)}")
    if result.reasoning:
        path = save_temp_text(REASONING_FILE_NAME, result.reasoning)
        if path:
            printt(f"{COL_DEFAULT}Reasoning log: {COL_DIM}{text_util.make_terminal_hyperlink(path, is_file=True)}")
    printt()

    if result.hit_output_limit:
        ask.ask_error(LlmRequestUtil.describe_output_limit(result))
        return
    if not result.reply.strip():
        # The provider didn't report why (otherwise caught above); a likely cause
        hint = (
            "\nThe model reasoned but never answered: it most likely hit its output-token limit "
            "while thinking (reasoning counts toward the limit)."
            if result.reasoning else ""
        )
        ask.ask_error(f"Empty reply from the LLM.{hint}")
        return

    try:
        normalized = RosterUtil.normalize_roster(AttributionReplyUtil.extract_json(result.reply))
    except Exception as e:
        error = f"Couldn't read a roster from the LLM's reply: {make_error_string(e)}"
        if result.finish_reason is None and not result.reply.rstrip().endswith(("}", "```")):
            error += "\nThe reply looks cut off (output-token limit?)."
        path = save_temp_text(FAILED_REPLY_FILE_NAME, result.reply)
        if path:
            error += f"\nThe reply was saved to: {path}"
        ask.ask_error(error)
        return
    for note in normalized.notes:
        printt(f"{COL_DIM}{note}")
    if normalized.notes:
        printt()

    roster_file = RosterFile(
        roster=normalized.roster,
        created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        llm=LlmRequestUtil.describe_llm(prefs),
        book_title=book.title,
        section_text_hashes=UtteranceUtil.section_text_hashes(book),
        raw_reply=result.reply,
    )
    error = SpeakerAssignFileUtil.save_roster_file(project_dir, roster_file)
    if error:
        ask.ask_error(error)
        return

    # A new roster invalidates attributions made against the previous one
    if has_attributions:
        error = SpeakerAssignFileUtil.delete_attributions_file(project_dir)
        if error:
            ask.ask_error(error)
            return
        printt(f"{COL_DIM}Discarded dialog assignments made with the previous roster")
        printt()

    print_accent_separator("Roster summary")
    if print_roster_info(state):
        ask.ask_enter_to_continue()


def print_roster_info(state: State) -> bool:
    """
    Prints the saved roster's user-facing (spoiler-free) fields. Aliases and
    descriptions are LLM-facing and may contain spoilers, so they're left out.
    Shows an error and returns False if the roster can't be loaded.
    """
    project_dir = state.project.dir_path
    result = SpeakerAssignFileUtil.load_roster_file(project_dir)
    if isinstance(result, str):
        ask.ask_error(result)
        return False
    roster_file = result
    roster = roster_file.roster

    provenance = [roster_file.created.replace("T", " ")[:16]] if roster_file.created else []
    if roster_file.llm.get("model"):
        provenance.append(f"model: {roster_file.llm['model']}")
    if provenance:
        printt(f"{COL_DIM}Created {', '.join(provenance)}")
    if RosterUtil.is_stale(roster_file.section_text_hashes, state.project.book):
        printt(f"{COL_ERROR}* {COL_DIM_ITALICS}The project text has changed since this roster was created")

    narration = roster.narration
    if narration != RosterNarration():
        person = f"{narration.person.value} person" if narration.person.value else "Unknown person"
        printt()
        printt(f"Narration: {person}")
        if narration.notes.strip():
            printt(f"{COL_DIM_ITALICS}{narration.notes.strip()}")

    id_width = max(len("id"), *(len(character.id) for character in roster.characters))
    printt()
    printt(f"{COL_DIM}{'#':>3}  {'id':<{id_width}}  {'tier':<10} {'gender':<7} name")
    for i, character in enumerate(roster.characters, 1):
        narrator = f" {COL_DIM}[narrator]" if character.is_narrator else ""
        tier = character.prominence.value or "?"
        printt(
            f"{COL_DIM}{i:>3}  {COL_ACCENT}{character.id:<{id_width}}{COL_DEFAULT}  "
            f"{tier:<10} {character.gender:<7} {character.name}{narrator}"
        )
    printt()

    path = SpeakerAssignFileUtil.roster_path(project_dir)
    printt(f"{COL_DEFAULT}Full roster data: "
           f"{COL_DIM}{text_util.make_terminal_hyperlink(path, is_file=True)} (may contain spoilers)")
    printt()
    return True


CANCEL_EVENT_FEEDBACK = {
    CancelEvent.PENDING:
        "Control-C pressed: will cancel as soon as the response starts "
        "(press again to cancel now)",
    CancelEvent.CANCELLING: "Control-C pressed, cancelling",
    CancelEvent.ABANDONING: "Control-C pressed again, cancelling now",
}

REASONING_FILE_NAME = "speaker_assign_roster_reasoning.txt"
FAILED_REPLY_FILE_NAME = "speaker_assign_roster_reply.txt"

def estimate_roster_input_tokens(book: Book) -> int:
    """Approximate input token count of the roster request (system prompt plus
    the message: instructions and the whole annotated book)."""
    return (len(ROSTER_SYSTEM_PROMPT) + len(RosterUtil.build_message(book))) // CHARS_PER_TOKEN


ROSTER_DESC = \
"""Sends the full book text to the LLM in a single request, which identifies
and saves the roster of speaking characters.

Requires an LLM with ~%1 context or more."""
