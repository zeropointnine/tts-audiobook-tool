"""Tests for NonWrappingOptionList's memoized option geometry."""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.app import App, ComposeResult
from textual.content import Content
from textual.visual import RichVisual

from tts_audiobook_tool.textual.textual_shared import NonWrappingOptionList
from textual.widgets import OptionList
from textual.widgets.option_list import Option


def run(coroutine):
    return asyncio.run(coroutine)


def make_options(count: int) -> list[Option]:
    """Build options with varied line counts like real editor rows."""
    options = []
    for index in range(count):
        prompt = Text(f"line {index} " * (1 + index % 3))
        options.append(Option(prompt, id=f"line-{index}"))
    return options


class ListApp(App[None]):
    def __init__(self, options: list[Option]):
        super().__init__()
        self.options_in = options

    def compose(self) -> ComposeResult:
        yield NonWrappingOptionList(*self.options_in, markup=False, compact=True)


class ComparisonApp(App[None]):
    """Mounts the memoized and base lists side by side for comparison."""

    def __init__(self, options: list[Option]):
        super().__init__()
        self.options_in = options

    def compose(self) -> ComposeResult:
        yield NonWrappingOptionList(*self.options_in, markup=False, compact=True)
        yield OptionList(*self.options_in, markup=False, compact=True)


async def exercise_layout_equality() -> None:
    app = ComparisonApp(make_options(30))
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        widget, control = app.query(OptionList)
        # Widths chosen to force multi-line wrapping for the longer prompts
        for width in (40, 20, 10):
            assert widget.get_content_height(None, None, width) == (
                control.get_content_height(None, None, width)
            )
        assert widget.get_content_height(None, None, 40) > 0


def test_heights_and_widths_match_plain_option_list() -> None:
    """Memoized geometry must produce identical layout to the base class."""
    run(exercise_layout_equality())


async def exercise_repeated_layout() -> int:
    calls = {"height": 0}
    visual_classes = (Content, RichVisual)
    original_height = {
        cls: cls.get_height for cls in visual_classes
    }

    def counting_height(self, rules, width):
        calls["height"] += 1
        return original_height.get(type(self), original_height[Content])(
            self, rules, width
        )

    for cls in visual_classes:
        cls.get_height = counting_height
    try:
        app = ListApp(make_options(25))
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            first_pass_heights = calls["height"]
            assert first_pass_heights > 0
            # Force additional layout passes
            for _ in range(3):
                app.query_one(OptionList).refresh(layout=True)
                await pilot.pause()
            assert calls["height"] == first_pass_heights
    finally:
        for cls in visual_classes:
            cls.get_height = original_height[cls]
    return calls["height"]


def test_layout_passes_measure_each_option_once() -> None:
    """Repeated layout passes must not re-measure memoized options."""
    assert run(exercise_repeated_layout()) > 0


async def exercise_prompt_replacement() -> int:
    app = ListApp(make_options(5))
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        option_list = app.query_one(NonWrappingOptionList)
        height_before = option_list.get_content_height(None, None, 40)

        option_list.replace_option_prompts(
            [(2, Text("tall\nrow\nwith\nfour\nlines"))], reflow=True
        )
        await pilot.pause()

        assert option_list.get_content_height(None, None, 40) > height_before
    return height_before


def test_prompt_replacement_invalidates_height() -> None:
    """Replacing a prompt with taller content must change the memoized height."""
    run(exercise_prompt_replacement())


async def exercise_reconcile() -> None:
    app = ListApp(make_options(4))
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        option_list = app.query_one(NonWrappingOptionList)
        height_before = option_list.get_content_height(None, None, 40)

        option_list.reconcile_options([("line-1", Text("one line"), True)])
        await pilot.pause()

        assert option_list.get_content_height(None, None, 40) != height_before


def test_reconcile_recomputes_changed_option_height() -> None:
    """reconcile_options must drop memo entries for replaced prompts."""
    run(exercise_reconcile())


async def exercise_resize() -> None:
    app = ListApp(make_options(10))
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        option_list = app.query_one(NonWrappingOptionList)
        wide_height = option_list.get_content_height(None, None, 80)
        narrow_height = option_list.get_content_height(None, None, 12)
        assert narrow_height > wide_height


def test_resize_recomputes_heights() -> None:
    """A width change must recompute memoized heights for the new width."""
    run(exercise_resize())
