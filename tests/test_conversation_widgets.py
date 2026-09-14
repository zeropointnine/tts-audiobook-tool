import asyncio

from rich.text import Text
from textual.app import App, ComposeResult

from tts_audiobook_tool.conversation.prompt_draft import PromptDraft
from tts_audiobook_tool.textual.conversation_widgets import (
    CONVERSATION_CSS,
    ConversationInputArea,
    ConversationRole,
    ConversationTextEditor,
    ConversationTranscript,
    SoundLevelMeter,
)
from tts_audiobook_tool.textual.textual_shared import (
    STYLE_DIM,
    TEXTUAL_SHARED_CSS,
)


class WidgetApp(App[None]):
    CSS = TEXTUAL_SHARED_CSS + CONVERSATION_CSS

    def __init__(
        self,
        *,
        selection_brackets: bool = False,
        microphone_input: bool = False,
        immediate_input: bool = False,
    ) -> None:
        super().__init__()
        self.draft = PromptDraft()
        self.submitted_values: list[str] = []
        self.selection_brackets = selection_brackets
        self.microphone_input = microphone_input
        self.immediate_input = immediate_input

    def on_conversation_text_editor_submitted(
        self, message: ConversationTextEditor.Submitted
    ) -> None:
        self.submitted_values.append(message.value)

    def compose(self) -> ComposeResult:
        yield ConversationTranscript()
        yield ConversationInputArea(
            self.draft,
            selection_brackets=self.selection_brackets,
            microphone_input=self.microphone_input,
            immediate_input=self.immediate_input,
        )


def run(coro) -> None:
    asyncio.run(coro)


def test_transcript_entries_are_semantic_and_update_in_place() -> None:
    async def exercise() -> None:
        app = WidgetApp()
        async with app.run_test(size=(50, 12)) as pilot:
            transcript = app.query_one(ConversationTranscript)
            you = transcript.append_entry(ConversationRole.YOU, "**literal**")
            llm = transcript.append_entry(ConversationRole.LLM, "...")
            system = transcript.append_entry(
                ConversationRole.SYSTEM, "\x1b[31merror\x1b[0m", parse_ansi=True
            )
            await pilot.pause()
            assert len(transcript.entries) == 3
            assert you.raw_content == "**literal**"
            transcript.update_entry(llm, Text("partial", style="dim"))
            await pilot.pause()
            assert llm.raw_content == "partial"
            assert len(transcript.entries) == 3
            transcript.remove_entry(system)
            await pilot.pause()
            assert len(transcript.entries) == 2

    run(exercise())


def test_input_area_is_fixed_three_rows_and_editor_submits_with_newlines() -> None:
    async def exercise() -> None:
        app = WidgetApp()

        async with app.run_test(size=(40, 10)) as pilot:
            area = app.query_one(ConversationInputArea)
            editor = app.query_one(ConversationTextEditor)
            area.show_text()
            assert area.prefix.display
            editor.insert("hello\r\nworld\n")
            await pilot.press("enter")
            await pilot.pause()
            assert app.submitted_values == ["hello\nworld"]
            assert area.size.height == 3
            assert (
                ConversationTextEditor.normalize_newlines("a\r\nb\rc")
                == "a\nb\nc"
            )

    run(exercise())


def test_sound_level_meter_formats_rounded_dbfs_and_right_growing_bar() -> None:
    assert SoundLevelMeter.format_level(None) == ""
    assert SoundLevelMeter.format_level(-5.4) == "████████  -5 dB"
    assert SoundLevelMeter.format_level(-14.6) == "  ██████ -15 dB"
    assert SoundLevelMeter.format_level(-40.0) == "       █ -40 dB"
    assert SoundLevelMeter.format_level(-45.0) == "         -45 dB"
    assert SoundLevelMeter.format_level(-200.0) == "         -99 dB"
    assert SoundLevelMeter.format_level(float("nan")) == "         -99 dB"
    assert SoundLevelMeter.format_level(3.0) == "████████   0 dB"
    assert SoundLevelMeter.format_level(-43.7)[:8] == "       ░"
    assert SoundLevelMeter.format_level(-42.4)[:8] == "       ▒"
    assert SoundLevelMeter.format_level(-41.2)[:8] == "       ▓"
    assert SoundLevelMeter.format_level(-38.7)[:8] == "      ░█"


def test_microphone_input_layout_omits_prefix_and_has_meter_margin() -> None:
    async def exercise() -> None:
        app = WidgetApp(microphone_input=True)
        async with app.run_test(size=(40, 10)) as pilot:
            area = app.query_one(ConversationInputArea)
            area.show_microphone()
            area.set_input_level(-5.0)
            await pilot.pause()
            assert area.size.height == 3
            assert not app.query_one("#conversation-input-prefix").display
            assert area.level_meter.display
            assert area.level_meter.content == "████████  -5 dB"
            assert area.level_meter.region.x - area.mic_view.region.right == 1

            area.show_disabled()
            assert not area.level_meter.display
            assert area.level_meter.content == ""

    run(exercise())


def test_immediate_microphone_input_is_one_row() -> None:
    async def exercise() -> None:
        app = WidgetApp(microphone_input=True, immediate_input=True)
        async with app.run_test(size=(40, 10)) as pilot:
            area = app.query_one(ConversationInputArea)
            area.show_microphone()
            await pilot.pause()
            assert area.size.height == 1
            assert area.mic_view.size.height == 1
            assert area.level_meter.size.height == 1

    run(exercise())


def test_manual_mic_draft_rendering_and_disabled_view() -> None:
    async def exercise() -> None:
        app = WidgetApp()
        async with app.run_test(size=(40, 10)) as pilot:
            area = app.query_one(ConversationInputArea)
            app.draft.add_phrase("first")
            app.draft.add_phrase("second")
            area.show_microphone()
            await pilot.pause()
            assert area.mic_view.display
            assert "first second" in str(area.mic_view.render())
            area.show_disabled()
            assert area.disabled_view.display
            assert not area.text_editor.display
            assert not area.prefix.display
            assert area.has_class("is-inactive")
            assert area.disabled_view.content == ""
            area.show_waiting()
            await pilot.pause()
            assert area.disabled_view.region.x == area.region.x
            assert area.has_class("is-waiting")
            waiting = area.disabled_view.content
            assert isinstance(waiting, Text)
            assert waiting.plain == "Please wait"
            assert str(waiting.style) == "italic"
            area.show_disabled()
            assert area.disabled_view.content == ""
            area.show_microphone()
            assert not area.has_class("is-inactive")

    run(exercise())


def test_mic_draft_selection_bracket_style() -> None:
    from rich.color import Color
    from rich.style import Style

    async def exercise() -> None:
        app = WidgetApp(selection_brackets=True)
        async with app.run_test(size=(40, 10)) as pilot:
            area = app.query_one(ConversationInputArea)
            app.draft.add_phrase("first")
            app.draft.add_phrase("second")
            area.show_microphone()
            await pilot.pause()
            content = area.mic_view.content
            assert isinstance(content, Text)
            # The highlight was on the last chunk, so it followed "second".
            assert content.plain == "first [second]"

            def style_at(position: int) -> Style | None:
                for start, end, style in content.spans:
                    if start <= position < end:
                        return (
                            Style.parse(style) if isinstance(style, str) else style
                        )
                return None

            # The selected phrase keeps the terminal's default color...
            selected = style_at(content.plain.find("second"))
            assert selected is not None
            assert selected.color == Color.default()
            assert not selected.italic
            # ...enclosed by dim square brackets; unselected phrases stay dim.
            for probe in ("first", "[", "]"):
                style = style_at(content.plain.find(probe))
                assert style is not None, probe
                assert style.color == Color.parse("#888888"), probe
                assert not style.italic, probe

    run(exercise())
