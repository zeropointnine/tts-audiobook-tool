import asyncio

import pytest
from textual.app import App
from textual.widgets import OptionList

from tts_audiobook_tool.textual.cut_point_selection_dialog import CutPointSelectionDialog


@pytest.mark.parametrize("has_split_points", [False, True])
def test_split_point_item_requires_defined_points(has_split_points: bool) -> None:
    from tts_audiobook_tool.textual.multi_select_dialog import (
        CUT_POINTS,
        MANUAL_ENTRY,
        MultiSelectDialog,
    )

    dialog = MultiSelectDialog([], [], has_split_points=has_split_points)
    assert dialog.trailing_results() == (
        [CUT_POINTS, MANUAL_ENTRY] if has_split_points else [MANUAL_ENTRY]
    )


def test_cut_point_ranges_match_displayed_line_numbers() -> None:
    dialog = CutPointSelectionDialog(232, {10, 49})
    assert dialog.labels == [
        "Before section marker 1 (lines 1-10)",
        "Section marker 1 up to section marker 2 (lines 11-49)",
        "Section marker 2 to end (lines 50-232)",
    ]
    assert dialog.ranges == [range(0, 10), range(10, 49), range(49, 232)]


@pytest.mark.parametrize(
    "points, expected",
    [([], [range(4)]), ([0, 1, 2, 3, 4], [range(i, i + 1) for i in range(4)])],
)
def test_cut_point_ranges_are_nonempty_and_cover_every_line(points, expected) -> None:
    # The implicit beginning/end do not create empty items; adjacent cuts are valid.
    dialog = CutPointSelectionDialog(4, points)
    assert dialog.ranges == expected
    if not points:
        assert dialog.labels == ["All lines (lines 1-4)"]


@pytest.mark.parametrize(
    "keys, expected",
    [
        (("down", "enter"), {10, 11}),
        (("end", "space"), {12}),
        (("escape",), None),
    ],
)
def test_cut_point_dialog_selects_or_cancels(keys, expected) -> None:
    # Enter and space return zero-based editor indices; cancellation returns None.
    async def exercise() -> None:
        app = App()
        results = []
        async with app.run_test() as pilot:
            app.push_screen(CutPointSelectionDialog(13, {10, 12}), results.append)
            await pilot.pause()
            await pilot.press(*keys)
            assert results == [expected]

    asyncio.run(exercise())


def test_cut_point_dialog_fits_content_scrolls_and_cancels_on_backdrop() -> None:
    # Long labels widen the box; tall lists scroll within the available screen.
    async def exercise() -> None:
        app = App()
        results = []
        async with app.run_test(size=(100, 16)) as pilot:
            dialog = CutPointSelectionDialog(10**12, range(10**12 - 30, 10**12))
            app.push_screen(dialog, results.append)
            await pilot.pause()
            box = dialog.query_one("#multi-select-dialog")
            assert box.region.width > 62
            assert box.region.height <= 16
            options = dialog.query_one(OptionList)
            await pilot.press("end")
            await pilot.pause()
            assert options.scroll_y > 0
            assert options.region.bottom <= box.content_region.bottom
            await pilot.click("#multi-select-title")
            assert app.screen is dialog
            await pilot.click(offset=(0, 0))
            assert results == [None]

    asyncio.run(exercise())
