from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM, COL_ERROR
from tts_audiobook_tool.project import Project


def make_blank_line_marker_indices(
    phrase_groups: list[PhraseGroup],
) -> list[int]:
    """Return indices of groups that begin after a detected blank-line break."""
    return [
        index + 1
        for index, phrase_group in enumerate(phrase_groups[:-1])
        if any(
            phrase.reason == Reason.SPACE_BREAK for phrase in phrase_group.phrases
        )
    ]


@dataclass(frozen=True)
class ClearSectionMarkers:
    """Dialog result requesting that all section markers be cleared."""

    marker_count: int


class SectionMarkersStep(Enum):
    """The current step of the section-markers dialog."""

    MENU = "menu"
    MANUAL = "manual"
    REGEX = "regex"
    BLANK_LINES = "blank_lines"
    CLEAR = "clear"


class SectionMarkersDialog(
    ModalScreen[list[int] | ClearSectionMarkers | None],
):
    """Offer additional section-marker options, in two steps."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("1", "choose_manual", "Manual", show=False),
        Binding("2", "choose_regex", "Regex", show=False),
        Binding("3", "choose_blank_lines", "Blank lines", show=False),
        Binding("4", "choose_clear_markers", "Clear markers", show=False),
        Binding("y", "confirm_step", "Confirm", show=False),
        Binding("n", "decline_step", "No", show=False),
    ]

    CSS = """
    SectionMarkersDialog {
        align: center middle;
        background: transparent;
    }

    #section-markers-dialog {
        width: 100%;
        min-width: 40;
        max-width: 80;
        height: auto;
        margin: 0 2;
        padding: 1 2;
        border: round #888888;
        background: ansi_default;
    }

    #section-markers-title,
    #section-markers-input,
    #section-markers-error,
    #section-markers-status,
    #section-markers-buttons {
        display: none;
    }

    #section-markers-title {
        text-align: center;
    }

    #section-markers-body.body-centered {
        text-align: center;
    }

    #section-markers-status {
        margin-top: 1;
        text-align: center;
    }

    #section-markers-buttons {
        height: 3;
        align-horizontal: center;
    }

    #section-markers-buttons Button {
        min-width: 16;
        margin: 0 1;
        color: #888888;
        background: ansi_default;
        border: round #888888;
        text-style: none;
    }

    #section-markers-buttons Button:hover,
    #section-markers-buttons Button:focus,
    #section-markers-buttons Button.-active {
        color: ansi_default;
        background: ansi_default;
        border: round #888888;
        text-style: none;
        background-tint: transparent;
        tint: transparent;
    }
    """

    def __init__(self, project: Project, marker_count: int) -> None:
        super().__init__()
        self.step = SectionMarkersStep.MENU
        self.project = project
        self.phrase_groups = project.phrase_groups
        self.line_count = len(self.phrase_groups)
        self.marker_count = marker_count
        self.blank_line_markers: list[int] = []

    def compose(self) -> ComposeResult:
        # Mark the input invisible (CSS visibility) before mounting: a hidden
        # input is still focusable, and it would otherwise grab the screen's
        # initial auto-focus and swallow the menu-step key bindings.
        input_widget = Input(id="section-markers-input", compact=True)
        input_widget.visible = False
        yield Vertical(
            Static("", id="section-markers-title", markup=False),
            Static("", id="section-markers-body", markup=False),
            input_widget,
            Static("", id="section-markers-error", markup=False),
            Static("", id="section-markers-status", markup=False),
            Horizontal(
                Button(Content.from_text("[Y]es", markup=False), id="yes"),
                Button(Content.from_text("[N]o", markup=False), id="no"),
                id="section-markers-buttons",
            ),
            id="section-markers-dialog",
        )

    def on_mount(self) -> None:
        self.update_dialog_content()

    def update_dialog_content(self) -> None:
        """Re-render the title and body for the current step."""
        title_text: Text
        body_text: Text
        status_text = Text("")
        show_input = False
        show_buttons = False
        show_status = False
        if self.step is SectionMarkersStep.MENU:
            title_text = Text.from_ansi(f"{COL_ACCENT}More options\n")
            body_lines = [
                "[1] Enter line number/s",
                "[2] Add using regular expression",
                "[3] Add at blank lines",
            ]
            if self.marker_count > 0:
                label = app_text.get_section_marker_label(
                    self.project, is_title_case=False
                )
                body_lines.append(f"[4] Clear {label}")
            body_text = Text.from_ansi("\n".join(body_lines))
        elif self.step is SectionMarkersStep.MANUAL:
            title_text = Text("")
            body_text = Text.from_ansi(
                f"Enter line number/s\n"
                f'{COL_DIM}Eg, "105, 200"'
            )
            show_input = True
        elif self.step is SectionMarkersStep.REGEX:
            title_text = Text.from_ansi("Enter a regular expression\n")
            label = app_text.get_section_marker_label(
                self.project, is_title_case=False, is_singular=True
            )
            body_text = Text.from_ansi(
                f'{COL_DIM}For example, if the source text has chapters that start with, '
                f'eg, "Chapter 241: Chapter Name", you could enter "Chapter \\d+",'
                f"{COL_DIM}which will insert a {label} for each line matching that pattern."
            )
            show_input = True
        elif self.step is SectionMarkersStep.CLEAR:
            title_text = Text("")
            label = app_text.get_section_marker_label(
                self.project,
                is_title_case=False,
                is_singular=self.marker_count == 1,
            )
            body_text = Text.from_ansi(f"Clear all {self.marker_count} {label}?")
            show_buttons = True
        else:
            title_text = Text("Add at blank lines\n")
            label = app_text.get_section_marker_label(
                self.project, is_title_case=False
            )
            body_text = Text.from_ansi(
                f'{COL_DIM}Adds {label} wherever 2+ consecutive blank lines '
                f'(ie, "\\n\\n\\n") are found in the source text.'
            )
            if not self.blank_line_markers:
                status_text = Text.from_ansi(f"{COL_ERROR}No matches found")
            else:
                status_text = Text(
                    f"Num matches found: {len(self.blank_line_markers)}"
                )
                show_buttons = True
            show_status = True
        title_widget = self.query_one("#section-markers-title", Static)
        title_widget.update(title_text)
        title_widget.display = bool(title_text.plain)
        body_widget = self.query_one("#section-markers-body", Static)
        body_widget.update(body_text)
        if self.step is SectionMarkersStep.CLEAR:
            body_widget.add_class("body-centered")
        else:
            body_widget.remove_class("body-centered")
        input_widget = self.query_one("#section-markers-input", Input)
        error_widget = self.query_one("#section-markers-error", Static)
        status_widget = self.query_one("#section-markers-status", Static)
        input_widget.display = show_input
        input_widget.visible = show_input
        error_widget.display = False
        error_widget.visible = False
        status_widget.update(status_text)
        status_widget.display = show_status
        status_widget.visible = show_status
        buttons_row = self.query_one("#section-markers-buttons", Horizontal)
        buttons_row.display = show_buttons
        buttons_row.visible = show_buttons
        if show_input:
            input_widget.value = ""
            error_widget.update("")
            self.call_after_refresh(lambda: input_widget.focus())
        elif show_buttons:
            self.call_after_refresh(
                lambda: self.query_one("#yes", Button).focus()
            )

    def parse_regex_input(self, value: str) -> list[int] | str:
        """Match the pattern against each phrase group and return zero-indexed markers.

        Returns the marker list, or an error string to display.
        """
        try:
            pattern = re.compile(value, flags=re.IGNORECASE)
        except re.error:
            return "Syntax error"
        markers = [
            index
            for index, phrase_group in enumerate(self.phrase_groups)
            if pattern.match(phrase_group.text)
        ]
        if not markers:
            return "No matches"
        if 0 in markers:
            markers.remove(0)
        return markers

    def parse_manual_input(self, value: str) -> list[int] | str:
        """Parse comma-delimited one-indexed line numbers into zero-indexed markers.

        Returns the marker list, or an error string to display.
        """
        one_indexed_items: list[int] = []
        for string_item in value.split(","):
            stripped = string_item.strip()
            try:
                one_indexed_items.append(int(stripped))
            except ValueError:
                return f"Parse error: {stripped}"
        one_indexed_items = sorted(set(one_indexed_items))
        for item in one_indexed_items:
            if item < 1 or item > self.line_count:
                return f"Index out of range: {item}"
        zero_indexed_items = [item - 1 for item in one_indexed_items]
        if 0 in zero_indexed_items:
            zero_indexed_items.remove(0)
        return zero_indexed_items

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "section-markers-input":
            return
        value = event.value.strip()
        if not value:
            self.dismiss(None)
            return
        if self.step is SectionMarkersStep.MANUAL:
            result = self.parse_manual_input(value)
        else:
            result = self.parse_regex_input(value)
        if isinstance(result, str):
            error_widget = self.query_one("#section-markers-error", Static)
            error_widget.update(Text.from_ansi(f"{COL_ERROR}{result}"))
            error_widget.display = True
            error_widget.visible = True
            return
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_choose_manual(self) -> None:
        if self.step is not SectionMarkersStep.MENU:
            return
        self.step = SectionMarkersStep.MANUAL
        self.update_dialog_content()

    def action_choose_regex(self) -> None:
        if self.step is not SectionMarkersStep.MENU:
            return
        self.step = SectionMarkersStep.REGEX
        self.update_dialog_content()

    def action_choose_blank_lines(self) -> None:
        if self.step is not SectionMarkersStep.MENU:
            return
        self.blank_line_markers = make_blank_line_marker_indices(self.phrase_groups)
        self.step = SectionMarkersStep.BLANK_LINES
        self.update_dialog_content()

    def action_choose_clear_markers(self) -> None:
        if self.step is not SectionMarkersStep.MENU:
            return
        if self.marker_count == 0:
            return
        self.step = SectionMarkersStep.CLEAR
        self.update_dialog_content()

    def action_confirm_step(self) -> None:
        if self.step is not SectionMarkersStep.CLEAR:
            return
        self.dismiss(ClearSectionMarkers(self.marker_count))

    def action_decline_step(self) -> None:
        if self.step is not SectionMarkersStep.CLEAR:
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes":
            if self.step is SectionMarkersStep.BLANK_LINES:
                if not self.blank_line_markers:
                    return
                self.dismiss(self.blank_line_markers)
            else:
                self.dismiss(ClearSectionMarkers(self.marker_count))
        elif event.button.id == "no":
            self.dismiss(None)
