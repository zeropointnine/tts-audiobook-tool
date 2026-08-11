import asyncio

from rich.text import Text
from textual.app import App
from textual.widgets import Button, Static

from tts_audiobook_tool.textual.alert_dialog import AlertDialog


def run(coroutine) -> None:
    asyncio.run(coroutine)


def test_alert_dialog_renders_optional_error_colored_title_and_copy() -> None:
    app = App[None]()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(
                AlertDialog(
                    title="Cannot generate audio",
                    copy="Choose a voice\nConfigure the model",
                )
            )
            await pilot.pause()

            title = app.screen.query_one("#alert-title", Static).content
            copy = app.screen.query_one("#alert-copy", Static).content
            assert isinstance(title, Text)
            assert isinstance(copy, Text)
            assert title.plain == "Cannot generate audio"
            assert copy.plain == "Choose a voice\nConfigure the model"
            assert title.spans and str(title.spans[0].style) == "color(196)"
            assert copy.spans and str(copy.spans[0].style) == "color(196)"
            assert app.screen.focused is app.screen.query_one("#ok", Button)

    run(exercise())


def test_alert_dialog_allows_omitted_title_and_copy_and_ok_dismisses() -> None:
    app = App[None]()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(AlertDialog())
            await pilot.pause()

            assert not app.screen.query("#alert-title")
            assert not app.screen.query("#alert-copy")
            await pilot.click(app.screen.query_one("#ok", Button))
            assert not isinstance(app.screen, AlertDialog)

    run(exercise())


def test_alert_dialog_escape_dismisses() -> None:
    app = App[None]()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(AlertDialog(copy="Blocked"))
            await pilot.press("escape")
            assert not isinstance(app.screen, AlertDialog)

    run(exercise())
