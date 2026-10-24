from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign import assign_menu, voices_menu
from tts_audiobook_tool.menus.speaker_assign.roster_menu import make_roster_stale_marker, roster_menu
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
        if has_dialog:
            requirement = f"{COL_DIM}Requires text imported with dialog segmentation enabled."
        elif book.segmentation_settings.dialog_segmentation:
            requirement = f"{COL_ERROR}No dialog found in the project text."
        else:
            requirement = ""
        subheading = SPEAKER_ASSIGN_DESC + ("\n\n" + requirement if requirement else "")

        def make_roster_label(_: State) -> str:
            if not book.segmentation_settings.dialog_segmentation:
                return f"Step 1: Create character roster {COL_DIM}(requires dialog segmentation)"
            if SpeakerAssignFileUtil.has_roster(state.project.dir_path):
                return (
                    f"Step 1: Create/view character roster {COL_DIM}(completed)"
                    + make_roster_stale_marker(state)
                )
            return "Step 1: Create character roster"

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
            return [
                MenuItem(make_roster_label, on_select_roster),
                MenuItem(assign_menu.make_step_label, lambda _, __: assign_menu.assign_menu(state)),
                MenuItem(voices_menu.make_step_label, lambda _, __: voices_menu.voices_menu(state)),
            ]

        MenuUtil.menu(
            state,
            heading="Assign character voices using LLM",
            items=items_maker,
            subheading=subheading,
            breadcrumb="Assign character voices using LLM",
        )


SPEAKER_ASSIGN_DESC = \
"""Uses an LLM to determine which character speaks each line of dialog,
so that main characters can each get their own voice.

Step 1 identifies the book's speaking characters.
Step 2 assigns a character to each line of dialog.
Step 3 is where you map characters to voices and apply them to the project."""
