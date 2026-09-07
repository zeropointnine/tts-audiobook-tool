"""Disposable probe: combined measurement-cache + fixed-width win."""

import asyncio
import time

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.textual.textual_shared import HangingIndentText, NonWrappingOptionList
from textual.app import App, ComposeResult
from textual.widgets.option_list import Option

N_GROUPS = 4000
COL_DIM = "\x1b[2m"
COL_DEFAULT = "\x1b[0m"

groups = []
for i in range(N_GROUPS):
    words = " ".join(f"word{i}_{j}" for j in range(12))
    groups.append(PhraseGroup([
        Phrase(words, Reason.SENTENCE),
        Phrase("Next sentence here.", Reason.PARAGRAPH),
    ]))

def format_line(i: int, g: PhraseGroup):
    prefix = f"{COL_DIM}[{i + 1:05d}] {COL_DEFAULT}"
    return HangingIndentText.from_ansi(
        ansi_text=f"{prefix}{g.presentable_text_lf_ansi}",
        content_start=7,
        max_lines=3,
        style="",
    )

class FastMeasuringOptionList(NonWrappingOptionList):
    """No horizontal extent + heights measured once per (option, width)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._height_memo: dict[int, tuple[int, int]] = {}

    def _get_visual(self, option):
        if getattr(option, "_fast_visual", None) is None:
            option._fast_visual = _MemoizedVisual(
                super()._get_visual(option), self._height_memo, id(option)
            )
        return option._fast_visual

    def get_content_width(self, container, viewport) -> int:
        return container.width

class _MemoizedVisual:
    """Caches get_height per width for one wrapped visual."""
    def __init__(self, visual, memo, key):
        self._visual = visual
        self._memo = memo
        self._key = key

    def get_height(self, rules, width):
        memo = self._memo.get(self._key)
        if memo is not None and memo[0] == width:
            return memo[1]
        height = self._visual.get_height(rules, width)
        self._memo[self._key] = (width, height)
        return height

    def get_optimal_width(self, rules, container_width):
        return 1  # unused: get_content_width overridden

    def __getattr__(self, name):
        return getattr(self._visual, name)

class ListApp(App[None]):
    def __init__(self, list_cls):
        super().__init__()
        self.list_cls = list_cls

    def compose(self) -> ComposeResult:
        yield self.list_cls(
            *(Option(format_line(i, g), id=f"line-{i}") for i, g in enumerate(groups)),
            markup=False, compact=True,
        )

async def run(label, list_cls):
    app = ListApp(list_cls)
    start = time.perf_counter()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
    print(f"{label:34s} {(time.perf_counter() - start) * 1000:9.1f} ms")

async def main():
    await run("baseline", NonWrappingOptionList)
    await run("height memo + fixed width", FastMeasuringOptionList)
    await run("baseline again", NonWrappingOptionList)

asyncio.run(main())
