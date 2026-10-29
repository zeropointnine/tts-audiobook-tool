"""Step 1: the full-book roster request (see plans/llm-speaker-attribution.md)."""
import datetime
from dataclasses import dataclass
import math
import os

from tts_audiobook_tool import app_support, ask, text_util
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign.characters_view import print_characters_info
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
from tts_audiobook_tool.speaker_assign.roster import RosterFile
from tts_audiobook_tool.speaker_assign.roster_util import ROSTER_SYSTEM_PROMPT, RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import (
    duration_string,
    make_error_string,
    make_hotkey_string,
    print_accent_separator,
    print_feedback,
    printt,
)


def roster_menu(state: State) -> None:

    # Computed once on entry (walks the whole book), not on every redraw
    tokens = estimate_roster_input_tokens(state.project.book)
    # Always in thousands, rounded up, as the context it needs
    subheading = ROSTER_DESC.replace(
        "%1", f"{COL_ACCENT}~{math.ceil(tokens / 1000):,}k{COL_DEFAULT}"
    )

    def items_maker(_: State) -> list[MenuItem]:
        if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
            return [MenuItem("Start", on_select_roster_start)]
        return [MenuItem("Start over", on_select_roster_start), MenuItem("Clear", on_select_roster_clear)]

    MenuUtil.menu(
        state,
        heading="Generate character roster",
        items=items_maker,
        subheading=subheading,
        breadcrumb="Generate character roster",
    )


def on_select_roster_start(state: State, _item: MenuItem) -> bool:
    if not check_llm_configured(state):
        return False
    # No new heading: the confirmation and the run print directly below the menu
    if not confirm_roster_replacement(state):
        return False
    # Back to the parent menu once the roster is made, for the next step
    return create_roster(state)


def confirm_roster_replacement(state: State) -> bool:
    """
    Asks to confirm when starting would replace a roster or discard the step 2
    and 3 data that depend on it, listing what would be lost. True to go ahead.
    """
    project_dir = state.project.dir_path
    has_roster = SpeakerAssignFileUtil.has_roster(project_dir)
    dependents = RosterDependents.find(project_dir)
    if not has_roster and not dependents:
        return True
    if has_roster:
        printt("A roster already exists and will be replaced.")
    dependents.print_warnings()
    printt()
    return ask.ask_confirm("")


def on_select_roster_clear(state: State, _item: MenuItem) -> None:
    # No new heading: the confirmation prints directly below the menu
    clear_roster(state)


def clear_roster(state: State) -> None:
    """Deletes the saved roster, and the step 2 and 3 data that depend on it, after confirmation."""
    project_dir = state.project.dir_path
    dependents = RosterDependents.find(project_dir)
    printt("The character roster will be deleted.")
    dependents.print_warnings()
    printt()
    if not ask.ask_confirm(""):
        return
    # Dependents first: they're meaningless without the roster they were made against
    error = dependents.delete(project_dir) or SpeakerAssignFileUtil.delete_roster_file(project_dir)
    if error:
        ask.ask_error(error)
        return
    print_feedback("Roster cleared")


@dataclass(frozen=True)
class RosterDependents:
    """
    Which saved data depends on the roster's character ids, and so is
    invalidated when the roster is cleared or replaced: step 2's dialog
    assignments and step 3's voice mappings.
    """
    has_attributions: bool
    has_voices: bool

    @staticmethod
    def find(project_dir: str) -> "RosterDependents":
        return RosterDependents(
            has_attributions=os.path.isfile(SpeakerAssignFileUtil.attributions_path(project_dir)),
            has_voices=os.path.isfile(SpeakerAssignFileUtil.voices_path(project_dir)),
        )

    def __bool__(self) -> bool:
        return self.has_attributions or self.has_voices

    def print_warnings(self) -> None:
        if self.has_attributions:
            printt(f"Dialog assignments from step 2 will be {COL_ERROR}discarded{COL_DEFAULT}.")
        if self.has_voices:
            printt(f"Voice mappings from step 3 will be {COL_ERROR}discarded{COL_DEFAULT}.")

    def describe(self) -> str:
        """What was discarded, eg "dialog assignments and voice mappings"."""
        parts = (["dialog assignments"] if self.has_attributions else []) + \
                (["voice mappings"] if self.has_voices else [])
        return " and ".join(parts)

    def delete(self, project_dir: str) -> str:
        """Returns an error string on failure, otherwise ""."""
        return (SpeakerAssignFileUtil.delete_attributions_file(project_dir)
                or SpeakerAssignFileUtil.delete_voices_file(project_dir))


# ---

def create_roster(state: State) -> bool:
    """
    The full-book roster request: sends the book to the LLM (control-c
    cancels), then saves the roster and prints it. Nothing is saved or
    deleted unless a valid roster comes back. Doesn't ask before replacing
    saved data (see confirm_roster_replacement). Returns True if a roster was saved.
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
        return False

    try:
        llm = LlmRequestUtil.make_session(prefs, ROSTER_SYSTEM_PROMPT)
    except ValueError as e:
        ask.ask_error(f"Can't use the LLM settings: {e}")
        return False

    message = RosterUtil.build_message(book)
    input_chars = len(ROSTER_SYSTEM_PROMPT) + len(message)

    print_accent_separator("Making request")
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

    def fail(message: str) -> None:
        # The request ran (not cancelled) but no roster came of it
        ask.ask_error(message)

    if result.cancelled:
        feedback = "Cancelled. Nothing was saved."
        if result.abandoned:
            feedback += (f"\n{COL_DIM}The request was abandoned before the response started; "
                         f"the LLM provider may still process it.")
        ask.ask_enter_to_continue(feedback)
        return False
    if result.error is not None:
        fail(LlmRequestUtil.describe_error(result.error, prefs))
        return False

    # A zero reasoning count isn't evidence of no reasoning (see LlmSession), so omit it
    reasoning_part = f"{len(result.reasoning):,} reasoning chars, " if result.reasoning else ""
    printt(f"{COL_DEFAULT}Done: {COL_DIM}{len(result.reply):,} answer chars, "
           f"{reasoning_part}elapsed: {duration_string(result.elapsed)}")
    if result.reasoning:
        path = save_temp_text(REASONING_FILE_NAME, result.reasoning)
        if path:
            printt(f"{COL_DEFAULT}Reasoning log: {COL_DIM}{text_util.make_terminal_hyperlink(path, is_file=True)}")
    printt()

    if result.hit_output_limit:
        fail(LlmRequestUtil.describe_output_limit(result))
        return False
    if not result.reply.strip():
        # The provider didn't report why (otherwise caught above); a likely cause
        hint = (
            "\nThe model reasoned but never answered: it most likely hit its output-token limit "
            "while thinking (reasoning counts toward the limit)."
            if result.reasoning else ""
        )
        fail(f"Empty reply from the LLM.{hint}")
        return False

    try:
        normalized = RosterUtil.normalize_roster(AttributionReplyUtil.extract_json(result.reply))
    except Exception as e:
        error = f"Couldn't read a roster from the LLM's reply: {make_error_string(e)}"
        if result.finish_reason is None and not result.reply.rstrip().endswith(("}", "```")):
            error += "\nThe reply looks cut off (output-token limit?)."
        path = save_temp_text(FAILED_REPLY_FILE_NAME, result.reply)
        if path:
            error += f"\nThe reply was saved to: {path}"
        fail(error)
        return False
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
        fail(error)
        return False

    # A new roster may have different character ids, invalidating the
    # attributions and voice mappings made with the previous one
    dependents = RosterDependents.find(project_dir)
    if dependents:
        error = dependents.delete(project_dir)
        if error:
            fail(error)
            return False
        printt(f"{COL_DIM}Discarded {dependents.describe()} made with the previous roster")
        printt()

    app_support.play_done_sound()
    print_accent_separator("Roster summary")
    if print_characters_info(state):
        ask.ask_enter_to_continue()
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
"""Sends the full book text to the LLM in a single request (%1 tokens),
which identifies and saves the roster of speaking characters."""
