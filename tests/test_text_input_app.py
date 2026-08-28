import asyncio

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Button, Rule, Static, TextArea

from tts_audiobook_tool.textual import text_input_app as text_input_app_module
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.text_input_app import (
    TextInputTextualApp,
    run_text_input_app,
)


def run(coroutine) -> None:
    asyncio.run(coroutine)


def test_app_composes_shared_header_divider_and_full_height_text_area(
    monkeypatch,
) -> None:
    monkeypatch.setattr(text_input_app_module.platform, "system", lambda: "Linux")
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)):
            header = app.query_one("#header", Vertical)
            first_line = app.query_one("#header-line-0", Static).content
            second_line = app.query_one("#header-line-1", Static).content
            third_line = app.query_one("#header-line-2", Static).content
            divider = app.query_one("#header-divider", Rule)
            text_area = app.query_one("#text-input", TextArea)

            assert isinstance(first_line, Text)
            assert first_line.plain == "Enter/paste text of any length"
            assert str(first_line.spans[0].style) == "#ffaa44"
            assert isinstance(second_line, Text)
            assert second_line.plain == (
                "Press [CTRL+SHIFT+V] to paste from the system clipboard"
            )
            assert str(second_line.spans[0].style) == "#888888"
            assert isinstance(third_line, Text)
            assert third_line.plain == "Press [ESC] to finish"
            assert any(
                span.start == 7
                and span.end == 10
                and str(span.style) == "#ffaa44"
                for span in third_line.spans
            )
            assert header.region.height == 3
            assert divider.region.height == 1
            assert text_area.region.height == 20
            assert text_area.has_focus
            assert not app.query("#status-bar")

    run(exercise())


def test_app_uses_command_v_clipboard_hint_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(text_input_app_module.platform, "system", lambda: "Darwin")
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test():
            line = app.query_one("#header-line-1", Static).content
            assert isinstance(line, Text)
            assert line.plain == (
                "Press [COMMAND+V] to paste from the system clipboard"
            )

    run(exercise())


def test_enter_adds_a_newline_without_finishing_input() -> None:
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("f", "i", "r", "s", "t", "enter")
            await pilot.press("s", "e", "c", "o", "n", "d")

            assert app.query_one("#text-input", TextArea).text == "first\nsecond"
            assert app.return_value is None
            await pilot.press("escape", "y")

    run(exercise())
    assert app.return_value == "first\nsecond"


def test_ctrl_a_selects_all_text() -> None:
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            text_area = app.query_one("#text-input", TextArea)
            text_area.text = "First line\nSecond line"

            await pilot.press("ctrl+a")
            assert text_area.selected_text == "First line\nSecond line"

            await pilot.press("x")
            assert text_area.text == "x"
            await pilot.press("escape", "n")

    run(exercise())


def test_escape_with_empty_text_exits_without_confirmation() -> None:
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#text-input", TextArea).text = "  \n "
            await pilot.press("escape")
            assert not isinstance(app.screen, SaveChangesDialog)

    run(exercise())
    assert app.return_value == ""


def test_escape_with_text_confirms_and_returns_stripped_multiline_text() -> None:
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.query_one("#text-input", TextArea).text = "  First line\nSecond line  "
            await pilot.press("escape")

            assert isinstance(app.screen, SaveChangesDialog)
            assert str(app.screen.query_one("#save-changes-copy-line-1", Static).render()) == (
                "Save changes?"
            )
            assert str(app.screen.query_one("#yes", Button).label) == "[Y]es"
            assert str(app.screen.query_one("#no", Button).label) == "[N]o"

            await pilot.press("y")

    run(exercise())
    assert app.return_value == "First line\nSecond line"


def test_no_discards_text_and_modal_escape_resumes_editing() -> None:
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            text_area = app.query_one("#text-input", TextArea)
            text_area.text = "Keep editing"
            await pilot.press("escape")
            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(app.screen, SaveChangesDialog)
            assert text_area.has_focus
            assert app.return_value is None

            await pilot.press("escape", "n")

    run(exercise())
    assert app.return_value == ""


def test_ctrl_q_does_not_close_app() -> None:
    app = TextInputTextualApp()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+q")
            assert app.return_value is None
            await pilot.press("escape")

    run(exercise())
    assert app.return_value == ""


def test_runner_falls_back_to_legacy_multiline_input(monkeypatch) -> None:
    monkeypatch.setattr(text_input_app_module, "can_textual", lambda: False)
    monkeypatch.setattr(
        text_input_app_module.ask, "ask_multiline", lambda: "fallback text"
    )

    assert run_text_input_app() == "fallback text"


def test_runner_reports_app_failure_and_returns_empty(monkeypatch) -> None:
    errors: list[str] = []

    class FailingApp:
        _exception = None

        def run(self, *, inline: bool):
            assert inline is False
            raise RuntimeError("broken terminal")

    monkeypatch.setattr(text_input_app_module, "can_textual", lambda: True)
    monkeypatch.setattr(text_input_app_module, "TextInputTextualApp", FailingApp)
    monkeypatch.setattr(text_input_app_module.ask, "ask_error", errors.append)

    assert run_text_input_app() == ""
    assert errors == ["RuntimeError: broken terminal"]
