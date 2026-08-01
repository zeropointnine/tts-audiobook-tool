from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.text_ops.text_edit_session import PhraseGroupSplitPoint


class PhraseGroupSplitDialog(ModalScreen[PhraseGroupSplitPoint | None]):
    """Choose a boundary between existing phrases in one phrase group."""

    AUTO_FOCUS = "#split-boundary"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    PhraseGroupSplitDialog {
        align: center middle;
        background: transparent;
    }

    #phrase-group-split-dialog {
        width: 76;
        height: auto;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #split-preview {
        height: auto;
        max-height: 8;
        margin-bottom: 1;
    }

    #split-error {
        height: 1;
        color: ansi_red;
    }
    """

    def __init__(self, phrase_group: PhraseGroup) -> None:
        super().__init__()
        self.phrase_group = phrase_group

    def compose(self) -> ComposeResult:
        boundary_count = len(self.phrase_group.phrases) - 1
        preview = "\n".join(
            f"[{index}] {phrase.presentable_text}"
            for index, phrase in enumerate(self.phrase_group.phrases, start=1)
        )
        yield Vertical(
            Static(
                "Split after which phrase?\n",
                markup=False,
            ),
            Static(preview, id="split-preview", markup=False),
            Input(
                placeholder=f"1-{boundary_count}",
                id="split-boundary",
                type="integer",
                compact=True,
            ),
            Static("", id="split-error", markup=False),
            id="phrase-group-split-dialog",
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "split-boundary":
            return
        try:
            boundary = int(event.value)
        except ValueError:
            boundary = 0
        if not 0 < boundary < len(self.phrase_group.phrases):
            self.query_one("#split-error", Static).update("Bad value")
            return
        self.dismiss(PhraseGroupSplitPoint(boundary))

    def action_cancel(self) -> None:
        self.dismiss(None)
