from __future__ import annotations

import math
from enum import Enum
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Static, TextArea

from tts_audiobook_tool.conversation.prompt_draft import PromptDraft
from tts_audiobook_tool.textual.textual_shared import (
    STYLE_ACCENT,
    STYLE_DEFAULT,
    STYLE_DIM,
    STYLE_ERROR,
    STYLE_OK,
)


CONVERSATION_CSS = """\
#conversation-transcript {
    height: 1fr;
    width: 100%;
    overflow-y: auto;
    scrollbar-gutter: stable;
    padding: 0;
}

ConversationEntry {
    height: auto;
    width: 100%;
    padding: 0;
}

ConversationEntry.entry-gap {
    margin-bottom: 1;
}

#input-divider {
    height: 1;
    margin: 0;
    color: $col-dim;
}

#conversation-input-area {
    height: 3;
    min-height: 3;
    max-height: 3;
    width: 100%;
    layout: horizontal;
    overflow: hidden;
}

#conversation-input-area.is-compact {
    height: 1;
    min-height: 1;
    max-height: 1;
}

#conversation-input-prefix {
    width: 5;
    height: 100%;
    color: $col-accent;
}

#conversation-input-area.is-microphone #conversation-input-prefix {
    display: none;
}

#conversation-input-area.is-inactive #conversation-input-prefix {
    color: $col-dim;
}

#conversation-text-input,
#conversation-mic-input,
#conversation-disabled-input {
    width: 1fr;
    height: 100%;
    padding: 0;
    border: none;
    background: ansi_default;
    background-tint: transparent;
}

#conversation-text-input:focus {
    border: none;
    background: ansi_default;
    background-tint: transparent;
}

#conversation-text-input .text-area--cursor-line {
    background: ansi_default;
}

#conversation-mic-input {
    overflow-y: auto;
}

#conversation-disabled-input {
    color: $col-dim;
}

/* "Please wait" during an interrupt: default color, set via Rich style,
   unlike the blank disabled view which stays dim. */
#conversation-input-area.is-waiting #conversation-disabled-input {
    color: ansi_default;
}

#conversation-input-level {
    width: 15;
    min-width: 15;
    height: 1;
    margin-left: 1;
    content-align: right top;
}
"""


class ConversationRole(str, Enum):
    YOU = "You"
    LLM = "LLM"
    ECHO = "Echo"
    SYSTEM = "System"


_ROLE_STYLES = {
    ConversationRole.YOU: STYLE_ACCENT,
    ConversationRole.LLM: STYLE_ACCENT,
    ConversationRole.ECHO: STYLE_ACCENT,
    ConversationRole.SYSTEM: STYLE_DIM,
}


class ConversationEntry(Static):
    """One semantic transcript entry with raw content kept separately."""

    def __init__(
        self,
        role: ConversationRole,
        content: str | Text,
        *,
        parse_ansi: bool = False,
        error: bool = False,
        gap: bool = False,
        id: str | None = None,
    ) -> None:
        self.role = role
        self.raw_content = content.plain if isinstance(content, Text) else content
        self.parse_ansi = parse_ansi
        self.error = error
        self.presentation = content.copy() if isinstance(content, Text) else None
        super().__init__(self.make_renderable(), id=id, markup=False)
        if gap:
            self.add_class("entry-gap")

    def make_renderable(self) -> Text:
        label_style = STYLE_ERROR if self.error else _ROLE_STYLES[self.role]
        if self.role is ConversationRole.SYSTEM:
            result = Text()
        else:
            result = Text(f"{self.role.value}: ", style=label_style)
        if self.presentation is not None:
            result.append_text(self.presentation.copy())
        elif self.parse_ansi:
            result.append_text(Text.from_ansi(self.raw_content))
        else:
            if self.error:
                style = STYLE_ERROR
            elif self.role is ConversationRole.SYSTEM:
                style = STYLE_DIM
            elif self.role is ConversationRole.YOU:
                style = STYLE_DIM
            else:
                style = STYLE_DEFAULT
            result.append(self.raw_content, style=style)
        return result

    def set_content(
        self,
        content: str | Text,
        *,
        parse_ansi: bool | None = None,
        error: bool | None = None,
    ) -> None:
        self.raw_content = content.plain if isinstance(content, Text) else content
        self.presentation = content.copy() if isinstance(content, Text) else None
        if parse_ansi is not None:
            self.parse_ansi = parse_ansi
        if error is not None:
            self.error = error
        self.update(self.make_renderable())


class ConversationTranscript(VerticalScroll):
    """Semantic entry list with detachable tail following."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "history_up", show=False),
        Binding("down", "history_down", show=False),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
    ]

    def __init__(self, *, id: str = "conversation-transcript") -> None:
        super().__init__(id=id, can_focus=True)
        self.entries: list[ConversationEntry] = []
        self.follow_tail = True

    def append_entry(
        self,
        role: ConversationRole,
        content: str | Text,
        *,
        parse_ansi: bool = False,
        error: bool = False,
        gap: bool = False,
        entry_id: str | None = None,
    ) -> ConversationEntry:
        entry = ConversationEntry(
            role,
            content,
            parse_ansi=parse_ansi,
            error=error,
            gap=gap,
            id=entry_id,
        )
        self.entries.append(entry)
        self.mount(entry)
        self._follow_after_update()
        return entry

    def remove_entry(self, entry: ConversationEntry) -> None:
        if entry in self.entries:
            self.entries.remove(entry)
        entry.remove()
        self._follow_after_update()

    def update_entry(self, entry: ConversationEntry, content: str | Text) -> None:
        if entry not in self.entries:
            return
        entry.set_content(content)
        self._follow_after_update()

    def attach_tail(self) -> None:
        self.follow_tail = True
        self.scroll_end(animate=False, immediate=True, x_axis=False)

    def _follow_after_update(self) -> None:
        if self.follow_tail:
            self.call_after_refresh(
                self.scroll_end, animate=False, immediate=True, x_axis=False
            )

    def action_history_home(self) -> None:
        self.follow_tail = False
        self.scroll_home(animate=False, x_axis=False)

    def action_history_up(self) -> None:
        self.follow_tail = False
        self.scroll_up(animate=False)

    def action_history_down(self) -> None:
        self.scroll_down(animate=False)
        self.call_after_refresh(self._reattach_if_at_end)

    def action_history_page_up(self) -> None:
        self.follow_tail = False
        self.scroll_page_up(animate=False)

    def action_history_page_down(self) -> None:
        self.scroll_page_down(animate=False)
        self.call_after_refresh(self._reattach_if_at_end)

    def action_history_end(self) -> None:
        self.scroll_end(animate=False, x_axis=False)
        self.follow_tail = True

    def _reattach_if_at_end(self) -> None:
        if self.scroll_offset.y >= self.max_scroll_y:
            self.follow_tail = True

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if self.follow_tail and new_value < self.max_scroll_y:
            self.follow_tail = False
        elif not self.follow_tail and new_value >= self.max_scroll_y:
            self.follow_tail = True

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.follow_tail = False
        super()._on_mouse_scroll_down(event)
        self.call_after_refresh(self._reattach_if_at_end)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.follow_tail = False
        super()._on_mouse_scroll_up(event)


class ConversationTextEditor(TextArea):
    """A wrapping two-row editor that preserves line feeds in pasted text."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    BINDINGS: ClassVar[list[BindingType]] = [
        *TextArea.BINDINGS,
        Binding("enter", "submit", show=False, priority=True),
    ]

    def __init__(self, *, id: str = "conversation-text-input") -> None:
        super().__init__(
            id=id,
            theme="vscode_dark",
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            tab_behavior="focus",
        )
        # The vscode_dark theme applies a solid editor background inline,
        # which overrides the CSS and paints a box around the input. Clear it
        # so the area renders on the terminal's default (unstyled) background.
        self.styles.background = None

    @staticmethod
    def normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_submit()
            return
        await super()._on_key(event)

    def action_submit(self) -> None:
        value = self.normalize_newlines(self.text).strip()
        if value:
            self.post_message(self.Submitted(value))

    async def _on_paste(self, event: events.Paste) -> None:
        event.stop()
        self.insert(self.normalize_newlines(event.text))

    def clear_text(self) -> None:
        self.load_text("")


class MicDraftView(Static):
    def __init__(
        self,
        draft: PromptDraft,
        *,
        selection_brackets: bool = False,
        id: str = "conversation-mic-input",
    ) -> None:
        self.draft = draft
        self.selection_brackets = selection_brackets
        super().__init__("", id=id, markup=False)
        self.refresh_draft()

    def refresh_draft(self) -> None:
        text = Text()
        if self.draft.is_empty:
            text.append("Speak into the mic...", style=f"italic {STYLE_OK}")
        else:
            for index, phrase in enumerate(self.draft.phrases):
                if index:
                    text.append(" ")
                if index == self.draft.selected_index:
                    if self.selection_brackets:
                        # "Microphone (submit by pressing ENTER)" mode: the
                        # selected phrase keeps the terminal's default color
                        # and is enclosed by dim square brackets.
                        text.append("[", style=STYLE_DIM)
                        text.append(phrase, style=STYLE_DEFAULT)
                        text.append("]", style=STYLE_DIM)
                    else:
                        text.append(phrase, style=f"italic {STYLE_ACCENT}")
                else:
                    text.append(phrase, style=STYLE_DIM)
        self.update(text)
        self.call_after_refresh(self.scroll_end, animate=False)


class SoundLevelMeter(Static):
    """Eight-cell, right-aligned microphone RMS meter."""

    BAR_WIDTH = 8
    DISPLAY_FLOOR_DB = -99
    BAR_FLOOR_DB = -45
    BAR_CEILING_DB = -5
    DB_PER_CELL = 5
    INTENSITY_CHARS = ("", "░", "▒", "▓")

    def __init__(self, id: str = "conversation-input-level") -> None:
        super().__init__("", id=id, markup=False)
        self._rendered_level = ""

    @classmethod
    def format_level(cls, level_db: float | None) -> str:
        if level_db is None:
            return ""
        if not math.isfinite(level_db):
            level_db = float(cls.DISPLAY_FLOOR_DB)
        rounded_db = max(cls.DISPLAY_FLOOR_DB, min(0, round(level_db)))
        bar_level_db = max(cls.BAR_FLOOR_DB, min(cls.BAR_CEILING_DB, level_db))
        intensity_steps = len(cls.INTENSITY_CHARS)
        intensity_units = int(
            math.floor(
                (bar_level_db - cls.BAR_FLOOR_DB)
                * intensity_steps
                / cls.DB_PER_CELL
                + 0.5
            )
        )
        intensity_units = min(cls.BAR_WIDTH * intensity_steps, intensity_units)
        full_cells, partial_intensity = divmod(intensity_units, intensity_steps)
        partial = cls.INTENSITY_CHARS[partial_intensity]
        bar = partial + "█" * full_cells
        return f"{bar:>{cls.BAR_WIDTH}}{rounded_db:>4} dB"

    def set_level(self, level_db: float | None) -> None:
        rendered = self.format_level(level_db)
        if rendered == self._rendered_level:
            return
        self._rendered_level = rendered
        self.update(rendered)


class ConversationInputArea(Horizontal):
    """Mode-aware shell which swaps text, mic, and blank views."""

    def __init__(
        self,
        draft: PromptDraft,
        *,
        selection_brackets: bool = False,
        microphone_input: bool = False,
        immediate_input: bool = False,
        id: str = "conversation-input-area",
    ) -> None:
        classes = "is-microphone" if microphone_input else ""
        if immediate_input:
            classes += " is-compact"
        super().__init__(id=id, classes=classes.strip() or None)
        self.draft = draft
        self.microphone_input = microphone_input
        self.prefix = Static("You:", id="conversation-input-prefix", markup=False)
        self.text_editor = ConversationTextEditor()
        self.mic_view = MicDraftView(draft, selection_brackets=selection_brackets)
        self.disabled_view = Static("", id="conversation-disabled-input", markup=False)
        self.level_meter = SoundLevelMeter()

    def compose(self) -> ComposeResult:
        yield self.prefix
        yield self.text_editor
        yield self.mic_view
        yield self.disabled_view
        yield self.level_meter

    def on_mount(self) -> None:
        self.show_disabled()

    def show_text(self, *, focus: bool = True) -> None:
        self.remove_class("is-inactive")
        self.prefix.display = True
        self.text_editor.display = True
        self.mic_view.display = False
        self.disabled_view.display = False
        self.level_meter.display = False
        self.level_meter.set_level(None)
        self.text_editor.disabled = False
        if focus:
            self.text_editor.focus()

    def show_microphone(self) -> None:
        self.remove_class("is-inactive")
        self.prefix.display = False
        self.text_editor.display = False
        self.mic_view.display = True
        self.disabled_view.display = False
        self.level_meter.display = self.microphone_input
        self.level_meter.set_level(None)
        self.mic_view.refresh_draft()

    def show_disabled(self) -> None:
        self.remove_class("is-waiting")
        self.add_class("is-inactive")
        self.prefix.display = False
        self.text_editor.display = False
        self.mic_view.display = False
        self.disabled_view.display = True
        self.level_meter.display = False
        self.level_meter.set_level(None)
        self.text_editor.disabled = True
        self.disabled_view.update("")

    def show_waiting(self) -> None:
        self.show_disabled()
        # The color comes from the is-waiting CSS rule (ansi_default); the
        # Rich style carries the italics.
        self.add_class("is-waiting")
        self.disabled_view.update(Text("Please wait", style="italic"))

    def set_input_level(self, level_db: float | None) -> None:
        if self.level_meter.display:
            self.level_meter.set_level(level_db)

    def clear(self) -> None:
        self.text_editor.clear_text()
        self.draft.reset()
        self.mic_view.refresh_draft()
