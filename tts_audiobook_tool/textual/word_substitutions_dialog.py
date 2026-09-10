from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM, COL_ERROR

ADD_TITLE = "Add item"
EDIT_TITLE = "Edit item"
ALREADY_EXISTS_MESSAGE = "Already exists"
WHITESPACE_MESSAGE = "Original word can't contain spaces"
EMPTY_KEY_MESSAGE = "Original word must contain a word"


def normalize_original_key(value: str) -> tuple[str, str]:
    """Return a matchable key plus an error message, if any.

    The matcher compares one whitespace-delimited token after peeling leading
    and trailing punctuation, so keys are normalized identically here. Internal
    whitespace can never match, and a key that is only punctuation has an empty
    comparison token, so both are rejected.
    """
    value = value.strip()
    if any(char.isspace() for char in value):
        return value, WHITESPACE_MESSAGE
    normalized = app_text.split_raw_word(value)[1]
    if not normalized:
        return value, EMPTY_KEY_MESSAGE
    return normalized, ""


@dataclass(frozen=True)
class WordSubstitutionEdit:
    """One committed add or edit from the word-substitutions dialog."""

    original: str
    substitution: str
    previous_original: str | None = None


class WordSubstitutionsDialog(ModalScreen[WordSubstitutionEdit | None]):
    """Edit one original/replacement pair before it is staged by the editor."""

    AUTO_FOCUS = "#original-input"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "toggle_focus", "Next field", show=False),
        Binding("shift+tab", "toggle_focus", "Previous field", show=False),
        Binding("down", "toggle_focus", "Next field", show=False),
        Binding("up", "toggle_focus", "Previous field", show=False),
    ]

    CSS = """
    WordSubstitutionsDialog {
        align: center middle;
        background: transparent;
    }

    #word-substitutions-dialog {
        width: 62;
        height: auto;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #word-substitutions-blank,
    #word-substitutions-error {
        height: 1;
    }

    #word-substitutions-title-row {
        width: 1fr;
        height: 1;
    }

    #word-substitutions-title {
        width: auto;
        height: 1;
    }

    #word-substitutions-submit-hint {
        width: 1fr;
        height: 1;
        text-align: right;
    }

    .word-substitutions-row {
        width: 1fr;
        height: 1;
    }

    .word-substitutions-row Input {
        width: 1fr;
    }

    .word-substitutions-label {
        width: 19;
        height: 1;
        padding: 0 1 0 0;
    }
    """

    def __init__(
        self,
        title: str,
        original: str,
        substitution: str,
        existing_keys: Iterable[str],
        previous_original: str | None = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.original = original
        self.substitution = substitution
        self.previous_original = previous_original
        # Editing a row without renaming its key must not collide with itself.
        previous_key = (
            None if previous_original is None else previous_original.casefold()
        )
        self.existing_keys = {
            key.casefold() for key in existing_keys if key.casefold() != previous_key
        }

    @classmethod
    def for_add(cls, existing_keys: Iterable[str]) -> "WordSubstitutionsDialog":
        """Build a blank dialog for appending a new substitution."""
        return cls(ADD_TITLE, "", "", existing_keys)

    @classmethod
    def for_edit(
        cls,
        original: str,
        substitution: str,
        existing_keys: Iterable[str],
    ) -> "WordSubstitutionsDialog":
        """Build a prefilled dialog which may rename the existing key."""
        return cls(
            EDIT_TITLE,
            original,
            substitution,
            existing_keys,
            previous_original=original,
        )

    def compose(self) -> ComposeResult:
        yield Vertical(
            Horizontal(
                Static(
                    Text.from_ansi(f"{COL_ACCENT}{self.title_text}"),
                    id="word-substitutions-title",
                    markup=False,
                ),
                Static(
                    Text.from_ansi(f"{COL_DIM}[ENTER] Submit"),
                    id="word-substitutions-submit-hint",
                    markup=False,
                ),
                id="word-substitutions-title-row",
            ),
            Static("", id="word-substitutions-blank", markup=False),
            Horizontal(
                Static(
                    Text.from_ansi(f"{COL_DIM}Original word:"),
                    id="word-substitutions-original-label",
                    markup=False,
                    classes="word-substitutions-label",
                ),
                Input(value=self.original, id="original-input", compact=True),
                classes="word-substitutions-row",
            ),
            Horizontal(
                Static(
                    Text.from_ansi(f"{COL_DIM}Substitution word:"),
                    id="word-substitutions-replacement-label",
                    markup=False,
                    classes="word-substitutions-label",
                ),
                Input(value=self.substitution, id="replacement-input", compact=True),
                classes="word-substitutions-row",
            ),
            Static("", id="word-substitutions-error", markup=False),
            id="word-substitutions-dialog",
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id not in ("original-input", "replacement-input"):
            return
        self.try_submit()

    def try_submit(self) -> None:
        """Normalize both fields and commit only a valid, effective change."""
        self.set_error("")
        original_input = self.query_one("#original-input", Input)
        replacement_input = self.query_one("#replacement-input", Input)
        original = original_input.value.strip()
        replacement = replacement_input.value.strip()
        original_input.value = original
        replacement_input.value = replacement

        if not original or not replacement:
            return
        original, error = normalize_original_key(original)
        original_input.value = original
        if error:
            self.set_error(error)
            return
        if original == replacement:
            return
        if original.casefold() in self.existing_keys:
            self.set_error(ALREADY_EXISTS_MESSAGE)
            return
        self.dismiss(
            WordSubstitutionEdit(
                original,
                replacement,
                previous_original=self.previous_original,
            )
        )

    def set_error(self, message: str) -> None:
        """Show or clear the dialog's single error line."""
        self.query_one("#word-substitutions-error", Static).update(
            Text.from_ansi(f"{COL_ERROR}{message}") if message else ""
        )

    def action_toggle_focus(self) -> None:
        """Move focus between the two fields on Tab, Shift+Tab, Up, or Down."""
        original_input = self.query_one("#original-input", Input)
        replacement_input = self.query_one("#replacement-input", Input)
        if self.focused is original_input:
            replacement_input.focus()
        else:
            original_input.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)
