from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import COL_DIM, COL_DIM_ITALICS, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign import assign_menu, voices_menu
from tts_audiobook_tool.menus.speaker_assign.characters_view import (
    VIEW_RESULTS_LABEL,
    characters_menu,
    make_roster_stale_marker,
)
from tts_audiobook_tool.menus.speaker_assign.roster_menu import roster_menu
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import print_feedback


class SpeakerAssignMenu:
    """
    Menu for LLM-based speaker attribution of dialog lines
    (see plans/llm-speaker-attribution.md).
    """

    @staticmethod
    def menu(state: State) -> None:

        if not state.project.phrase_groups:
            print_feedback("Requires source text", is_error=True)
            return

        # Computed once on entry (walks the whole book); the project text
        # can't change while in this menu
        book = state.project.book
        has_dialog = bool(UtteranceUtil.find_utterances(book, 0, len(book.flat_phrase_groups)))
        if not has_dialog and book.segmentation_settings.dialog_segmentation:
            requirement = f"{COL_ERROR}No dialog found in the project text."
        else:
            requirement = ""
        subheading = SPEAKER_ASSIGN_DESC + ("\n\n" + requirement if requirement else "")

        def make_roster_label(_: State) -> str:
            if not book.segmentation_settings.dialog_segmentation:
                return f"Step 1: Generate character roster {COL_DIM}(requires dialog segmentation)"
            if SpeakerAssignFileUtil.has_roster(state.project.dir_path):
                marker = make_roster_stale_marker(state)
                return f"Step 1: Generate character roster {COL_DIM}(completed{marker}{COL_DIM})"
            return "Step 1: Generate character roster"

        def on_select_roster(_: State, _item: MenuItem) -> None:
            if not book.segmentation_settings.dialog_segmentation:
                s = (
                    f"{COL_ERROR}The text was imported without dialog segmentation.\n"
                    "Re-import it with dialog segmentation enabled (Text > Segmentation settings).\n"
                )
                ask.ask_enter_to_continue(s)
                return
            roster_menu(state)

        def items_maker(_: State) -> list[MenuItem]:
            items = [
                MenuItem(make_roster_label, on_select_roster),
                MenuItem(assign_menu.make_step_label, lambda _, __: assign_menu.assign_menu(state)),
                MenuItem(voices_menu.make_step_label, lambda _, __: voices_menu.voices_menu(state)),
            ]
            if SpeakerAssignFileUtil.has_roster(state.project.dir_path):
                items.append(MenuItem(
                    lambda _: VIEW_RESULTS_LABEL + make_roster_stale_marker(state),
                    lambda _, __: characters_menu(state),
                    blank_line_before=True,
                ))
            return items

        MenuUtil.menu(
            state,
            heading="Auto-assign character voices",
            items=items_maker,
            subheading=subheading,
            breadcrumb="Auto-assign character voices",
        )


SPEAKER_ASSIGN_DESC = \
f"""Uses LLM to automatically determine which character speaks each line of dialog,
so that main characters can each get their own voice.

{COL_DIM_ITALICS}* A capable LLM with plenty of context and reasoning enabled is recommended.
{COL_DIM_ITALICS}* The full book text is sent to the configured LLM provider, so the usual
  considerations about token costs and privacy apply.
"""
