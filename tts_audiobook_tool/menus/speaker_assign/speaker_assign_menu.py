from tts_audiobook_tool.constants import COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.speaker_assign import assign_menu, voices_menu
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
        subheading = SPEAKER_ASSIGN_DESC + "\n\n" + (
            f"{COL_DIM}Requires text imported with dialog segmentation enabled."
            if has_dialog else
            f"{COL_ERROR}No dialog found in the project text: re-import with dialog segmentation enabled\n"
            f"(Text > Segmentation settings)."
        )

        def make_roster_label(_: State) -> str:
            label = "Step 1: Create character roster"
            if SpeakerAssignFileUtil.has_roster(state.project.dir_path):
                label += f" {COL_DIM}(completed)"
            return label

        def items_maker(_: State) -> list[MenuItem]:
            return [
                MenuItem(make_roster_label, lambda _, __: roster_menu(state)),
                MenuItem(assign_menu.make_step_label, lambda _, __: assign_menu.assign_menu(state)),
                MenuItem(voices_menu.make_step_label, lambda _, __: voices_menu.voices_menu(state)),
            ]

        MenuUtil.menu(
            state,
            heading="Auto-assign character voices",
            items=items_maker,
            subheading=subheading,
            breadcrumb="Auto-assign character voices",
        )


SPEAKER_ASSIGN_DESC = \
"""Uses an LLM to work out which character speaks each line of dialog, so that
main characters can each get their own voice, with a shared voice for the rest.

Step 1 identifies the book's speaking characters, step 2 assigns a character
to each line of dialog, and step 3 maps characters to voices. Only step 3
changes the project."""
