import asyncio

import pytest
from textual.app import App
from textual.widgets import Input, Static

from tts_audiobook_tool.textual.word_substitutions_dialog import (
    EMPTY_KEY_MESSAGE,
    WHITESPACE_MESSAGE,
    WordSubstitutionEdit,
    WordSubstitutionsDialog,
    normalize_original_key,
)


def run(coroutine) -> None:
    asyncio.run(coroutine)


def screen_text(dialog: WordSubstitutionsDialog, widget_id: str) -> str:
    return str(dialog.query_one(f"#{widget_id}", Static).render())


def test_add_dialog_is_blank_and_edit_dialog_is_prefilled() -> None:
    app = App[None]()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add(["Existing"]))
            await pilot.pause()
            assert screen_text(app.screen, "word-substitutions-title") == "Add item"
            assert app.screen.query_one("#original-input", Input).value == ""
            assert app.screen.query_one("#replacement-input", Input).value == ""
            assert screen_text(app.screen, "word-substitutions-error") == ""

            app.push_screen(
                WordSubstitutionsDialog.for_edit("Ariekei", "AriaKay", ["Ariekei"])
            )
            await pilot.pause()
            assert screen_text(app.screen, "word-substitutions-title") == "Edit item"
            assert app.screen.query_one("#original-input", Input).value == "Ariekei"
            assert app.screen.query_one("#replacement-input", Input).value == "AriaKay"

    run(exercise())


def test_tab_toggles_focus_between_the_two_inputs() -> None:
    app = App[None]()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]))
            await pilot.pause()
            original_input = app.screen.query_one("#original-input", Input)
            replacement_input = app.screen.query_one("#replacement-input", Input)
            assert app.screen.focused is original_input

            await pilot.press("tab")
            assert app.screen.focused is replacement_input

            await pilot.press("shift+tab")
            assert app.screen.focused is original_input

            await pilot.press("tab")
            assert app.screen.focused is replacement_input

    run(exercise())


def test_empty_or_equal_input_is_silently_ignored_after_stripping() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()
            original_input = app.screen.query_one("#original-input", Input)
            replacement_input = app.screen.query_one("#replacement-input", Input)

            original_input.value = "same"
            replacement_input.value = "  same  "
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert original_input.value == "same"
            assert replacement_input.value == "same"
            assert screen_text(app.screen, "word-substitutions-error") == ""

            replacement_input.value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert original_input.value == "same"
            assert replacement_input.value == ""

            original_input.value = ""
            replacement_input.value = "value"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert results == []

    run(exercise())


def test_valid_input_dismisses_with_stripped_values() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()
            app.screen.query_one("#original-input", Input).value = "  Ariekei  "
            app.screen.query_one("#replacement-input", Input).value = "  AriaKay  "

            await pilot.press("enter")
            await pilot.pause()

            assert results == [WordSubstitutionEdit("Ariekei", "AriaKay")]

    run(exercise())


def test_existing_key_shows_error_and_keeps_dialog_open() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(
                WordSubstitutionsDialog.for_add(["Ariekei"]), results.append
            )
            await pilot.pause()
            app.screen.query_one("#original-input", Input).value = "ARIEKEI"
            app.screen.query_one("#replacement-input", Input).value = "AriaKay"

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "Already exists" in screen_text(
                app.screen, "word-substitutions-error"
            )
            assert results == []

    run(exercise())


def test_editing_without_renaming_does_not_collide_with_its_own_key() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(
                WordSubstitutionsDialog.for_edit(
                    "Ariekei", "AriaKay", ["Ariekei", "Kilohour"]
                ),
                results.append,
            )
            await pilot.pause()
            app.screen.query_one("#replacement-input", Input).value = "Aria-Kay"

            await pilot.press("enter")
            await pilot.pause()

            assert results == [
                WordSubstitutionEdit(
                    "Ariekei", "Aria-Kay", previous_original="Ariekei"
                )
            ]

    run(exercise())


def test_escape_dismisses_none() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(app.screen, WordSubstitutionsDialog)
            assert results == [None]

    run(exercise())


@pytest.mark.parametrize(
    ("value", "expected_key", "expected_error"),
    [
        ("Ariekei", "Ariekei", ""),
        ("Higgs-Boson", "Higgs-Boson", ""),
        ("  e.g.  ", "e.g", ""),
        ("'tis", "tis", ""),
        ("U.S.A.", "U.S.A", ""),
        ("kilo hour", "kilo hour", WHITESPACE_MESSAGE),
        ("kilo\thour", "kilo\thour", WHITESPACE_MESSAGE),
        ("kilo\u00a0hour", "kilo\u00a0hour", WHITESPACE_MESSAGE),
        ("...", "...", EMPTY_KEY_MESSAGE),
        ("!!!", "!!!", EMPTY_KEY_MESSAGE),
    ],
)
def test_normalize_original_key(
    value: str, expected_key: str, expected_error: str
) -> None:
    assert normalize_original_key(value) == (expected_key, expected_error)


def test_internal_whitespace_key_shows_error_and_keeps_dialog_open() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()
            original_input = app.screen.query_one("#original-input", Input)
            original_input.value = "kilo hour"
            app.screen.query_one("#replacement-input", Input).value = "kilo-hour"

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "spaces" in screen_text(app.screen, "word-substitutions-error")
            assert original_input.value == "kilo hour"
            assert results == []

    run(exercise())


def test_key_punctuation_is_normalized_and_clears_a_stale_error() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()
            original_input = app.screen.query_one("#original-input", Input)
            replacement_input = app.screen.query_one("#replacement-input", Input)

            # An un-matchable key raises an error message.
            original_input.value = "kilo hour"
            replacement_input.value = "kilo-hour"
            await pilot.press("enter")
            await pilot.pause()
            assert "spaces" in screen_text(app.screen, "word-substitutions-error")

            # A silently-ignored submit clears the stale message.
            replacement_input.value = ""
            await pilot.press("enter")
            await pilot.pause()
            assert screen_text(app.screen, "word-substitutions-error") == ""

            # Then a punctuation-padded key normalizes and commits.
            original_input.value = "  e.g.  "
            replacement_input.value = "for example"
            await pilot.press("enter")
            await pilot.pause()

            assert results == [WordSubstitutionEdit("e.g", "for example")]

    run(exercise())


def test_punctuation_only_key_shows_error() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()
            app.screen.query_one("#original-input", Input).value = "..."
            app.screen.query_one("#replacement-input", Input).value = "dot dot dot"

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert "word" in screen_text(app.screen, "word-substitutions-error")
            assert results == []

    run(exercise())


def test_equality_is_checked_after_key_normalization() -> None:
    app = App[None]()
    results: list[WordSubstitutionEdit | None] = []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.push_screen(WordSubstitutionsDialog.for_add([]), results.append)
            await pilot.pause()
            app.screen.query_one("#original-input", Input).value = "same."
            app.screen.query_one("#replacement-input", Input).value = "same"

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, WordSubstitutionsDialog)
            assert screen_text(app.screen, "word-substitutions-error") == ""
            assert results == []

    run(exercise())
