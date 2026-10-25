from rich.color import Color
from rich.console import Console
from rich.text import Text
from textual.app import App
from textual.widgets import Static

from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM
from tts_audiobook_tool.textual.voice_select_dialog import (
    VoiceSelectDialog,
    compute_voice_grid_layout,
    make_voice_select_title,
)
from textual_editor_stubs import run


def layout(voice_count: int, screen_height: int, screen_width: int = 200):
    # Entries 17 wide; 4 chrome rows; content height = screen - 4 margin - 4 frame
    return compute_voice_grid_layout(
        voice_count=voice_count,
        entry_width=17,
        min_content_width=24,
        chrome_rows=4,
        screen_width=screen_width,
        screen_height=screen_height,
    )


def test_title_names_single_line_else_counts_lines() -> None:
    assert make_voice_select_title([17]) == "Line 17"
    assert make_voice_select_title([3, 4, 9]) == "3 lines selected"


def test_layout_grows_down_then_into_balanced_columns() -> None:
    # Room for 13 rows (25 - 4 - 4 - 4): up to 13 voices stay in one column...
    assert (layout(13, 25).rows, layout(13, 25).columns) == (13, 1)
    assert layout(13, 25).height == 13 + 4 + 4
    # ...and 14 split evenly into two columns of 7 rather than 13 + 1
    assert (layout(14, 25).rows, layout(14, 25).columns) == (7, 2)
    assert layout(14, 25).width == 17 * 2 + 4 + 6
    assert (layout(30, 25).rows, layout(30, 25).columns) == (10, 3)


def test_layout_is_capped_and_omits_columns_that_dont_fit() -> None:
    # 50 columns of content width (60 - 4 - 6) fit two 17-wide columns + gap
    result = layout(30, 25, screen_width=60)
    assert (result.columns, result.visible_columns) == (3, 2)
    assert result.width <= 60 - 4
    # Few voices: the width is set by the title/input rather than the list
    assert layout(2, 25).width == 24 + 6


class DialogApp(App):
    def __init__(self, dialog: VoiceSelectDialog) -> None:
        super().__init__()
        self.dialog = dialog
        self.results: list[int | None] = []

    def on_mount(self) -> None:
        self.push_screen(self.dialog, self.results.append)


def list_text(dialog: VoiceSelectDialog) -> Text:
    assert dialog.grid_layout is not None
    return dialog.list_text(dialog.grid_layout)


def ansi_color(code: str) -> Color | None:
    return Text.from_ansi(f"{code}x").get_style_at_offset(Console(), 0).color


def color_at(text: Text, substring: str) -> Color | None:
    return text.get_style_at_offset(Console(), text.plain.index(substring)).color


def test_entries_align_names_and_style_current_and_unnamed_voices() -> None:
    # Numbers pad to the widest ("[1]  " vs "[10] "), names over 12 characters
    # are cut, an unnamed voice reads "unnamed" (dim), and the current voice's
    # whole entry is accented.
    names = ["haru", "", "averylongname-xyz", *[f"v{n}" for n in range(4, 11)]]
    dialog = VoiceSelectDialog("Line 17", names, current_voice_index=0)
    app = DialogApp(dialog)

    async def exercise() -> None:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            text = list_text(dialog)
            lines = text.plain.split("\n")
            assert lines[:3] == ["[1]  haru", "[2]  unnamed", "[3]  averylongnam"]
            assert lines[9] == "[10] v10"
            accent = ansi_color(COL_ACCENT)
            assert color_at(text, "[1]") == accent
            assert color_at(text, "haru") == accent
            assert color_at(text, "unnamed") == ansi_color(COL_DIM)
            assert color_at(text, "[2]") != accent

    run(exercise())


def test_input_accepts_valid_number_rejects_out_of_range_and_blank_cancels() -> None:
    dialog = VoiceSelectDialog("Line 1", ["a", "b", "c"])
    app = DialogApp(dialog)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x", "9", "enter")  # letters are ignored
            assert app.screen is dialog
            error = dialog.query_one("#voice-select-error", Static)
            assert error.display
            assert "1 to 3" in str(error.render())
            await pilot.press("backspace", "3", "enter")
            assert app.results == [2]

    run(exercise())

    for keys in (("enter",), ("2", "escape")):
        app = DialogApp(VoiceSelectDialog("Line 1", ["a", "b"]))

        async def cancel(app=app, keys=keys) -> None:
            async with app.run_test() as pilot:
                await pilot.press(*keys)
                assert app.results == [None]

        run(cancel())
