"""Tests for the shared worker output area (`WorkerLogContentArea`).

The log's document always ends with the *current line*: in-progress
worker output (a dynamic progress bar, a status) lives on the document's
last line and updates in place, exactly as it would in a conventional
console. Committing the line (a newline) turns it into a normal history
line and starts a fresh empty current line below it, so a commit
advances the document by one line in the normal document flow — and
the log area never resizes.
"""

import asyncio

from textual import events
from textual.app import App, ComposeResult
from textual.events import Resize
from textual.geometry import Size

from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.textual.worker_content import WorkerLog, WorkerLogContentArea


class _HostApp(App[None]):
    """Minimal screen giving the content area a full layout to live in."""

    def __init__(self, output_filters: list[str] | None = None) -> None:
        super().__init__()
        self.output_filters = output_filters

    def compose(self) -> ComposeResult:
        yield WorkerLogContentArea(
            output_filters=self.output_filters,
            id="area",
        )


def run(coroutine) -> None:
    asyncio.run(coroutine)


def doc_lines(log: WorkerLog) -> list[str]:
    """The document's logical lines (plain text), in order."""
    return [line.text.plain for line in log._lines]


def test_current_line_updates_in_place_and_commit_advances_by_one_line() -> None:
    """The in-progress line is the document's last line: ticks replace
    it in place, a newline commits it to history and starts a fresh
    empty current line below it."""

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(100, 24)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()
            # The document starts with its (empty) current line.
            assert doc_lines(log) == [""]
            assert log.virtual_size.height == 1

            # A bar appears on the current line and ticks in place.
            area.feed([], "loading model 10%")
            area.feed([], "loading model 20%")
            await pilot.pause()
            assert doc_lines(log) == ["loading model 20%"]
            assert log.virtual_size.height == 1

            # Newlines commit: each becomes a normal history line, and a
            # fresh empty current line always ends the document.
            area.feed(["loading model 20%"], "")
            area.feed(["done"], "")
            await pilot.pause()
            assert doc_lines(log) == ["loading model 20%", "done", ""]
            assert log.virtual_size.height == 3

    run(exercise())


def test_app_lines_commit_the_current_line_to_history() -> None:
    """An app line (e.g. a cancellation notice) interrupts an active bar:
    the bar becomes committed history, the line lands below it, and the
    next console chunk continues on the fresh current line — the exact
    order a conventional console would show."""

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(100, 24)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            area.feed([], "bar 40%")
            area.append_lines(["Generation cancelled."])
            await pilot.pause()
            assert doc_lines(log) == ["bar 40%", "Generation cancelled.", ""]

            # The worker's next tick lands on the new current line.
            area.feed(["bar 50%"], "")
            assert doc_lines(log) == [
                "bar 40%",
                "Generation cancelled.",
                "bar 50%",
                "",
            ]

    run(exercise())


def test_finalize_commits_the_current_line() -> None:
    """At session end the current line is committed like a newline;
    finalizing an empty current line is a no-op."""

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(100, 24)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            area.feed([], "bar 100%")
            area.finalize()
            await pilot.pause()
            assert doc_lines(log) == ["bar 100%", ""]

            area.finalize()
            assert doc_lines(log) == ["bar 100%", ""]

    run(exercise())


def test_output_filters_discard_worker_lines_only_when_committed() -> None:
    """Filtered worker output can be shown live, but it never enters history
    through a newline, an app-line interruption, or finalization."""

    async def exercise() -> None:
        app = _HostApp(output_filters=["smem_size"])
        async with app.run_test(size=(100, 24)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            # A matching line remains visible while it is the live line.
            area.feed([], "kernel smem_size=123")
            assert doc_lines(log) == ["kernel smem_size=123"]

            # A newline discards that line while retaining ordinary output.
            area.feed(["kernel smem_size=123", "compiled"], "")
            assert doc_lines(log) == ["compiled", ""]

            # App-generated lines remain visible, but a matching worker line
            # that they interrupt does not become history.
            area.feed([], "another smem_size report")
            area.append_lines(["Generation cancelled."])
            assert doc_lines(log) == [
                "compiled",
                "Generation cancelled.",
                "",
            ]

            # Finalizing also drops a matching live worker line.
            area.feed([], "final smem_size report")
            area.finalize()
            assert doc_lines(log) == [
                "compiled",
                "Generation cancelled.",
                "",
            ]

    run(exercise())


def test_live_line_lifecycle_never_resizes_the_log_area() -> None:
    """A live line appearing, ticking, committing, and being interrupted
    must not change the log area's height: the current line is part of
    the document, so the area is a fixed region whose content scrolls.

    (The old implementation toggled a separate one-row chrome line in
    and out of the layout, which resized the `1fr` log by one row every
    time a dynamic progress bar started or finished, visibly resizing
    the content area and flickering its visible content.)
    """

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(100, 24)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            heights = [log.scrollable_content_region.height]
            area.feed([], "loading model 10%")
            await pilot.pause()
            heights.append(log.scrollable_content_region.height)
            area.feed([], "loading model 20%")
            await pilot.pause()
            heights.append(log.scrollable_content_region.height)
            area.feed(["loading model 20%"], "")
            await pilot.pause()
            heights.append(log.scrollable_content_region.height)
            area.append_lines(["Generation completed."])
            await pilot.pause()
            heights.append(log.scrollable_content_region.height)
            area.finalize()
            await pilot.pause()
            heights.append(log.scrollable_content_region.height)
            assert len(set(heights)) == 1

    run(exercise())


def test_long_line_wraps_and_reflows_on_resize() -> None:
    """A single long word wraps hard at the live content-region width;
    resizing the window reflows the wrapped line without losing content."""

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(40, 12)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            word = "x" * 250
            area.feed([], word)
            await pilot.pause()

            def expected_rows(width: int) -> int:
                # A single unbreakable word fills each row to the edge.
                return -(-len(word) // width)

            width = log.scrollable_content_region.width
            assert log.virtual_size.height == expected_rows(width)

            # Widen the window: the same word reflows to fewer rows.
            app.post_message(
                Resize(Size(200, 12), Size(200, 12), Size(200, 12))
            )
            await pilot.pause(0.05)
            width = log.scrollable_content_region.width
            assert width > 40
            assert log.virtual_size.height == expected_rows(width)
            assert doc_lines(log) == [word]

    run(exercise())


def test_osc_8_file_link_remains_clickable_across_wrapped_rows() -> None:
    """Rich link spans survive ANSI parsing and every hard-wrapped fragment."""

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(24, 12)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            filename = "a" * 80 + ".flac"
            url = f"/tmp/{filename}"
            link = make_terminal_hyperlink(url, filename, is_file=True)
            area.feed([f"Saved: {link}"], "")
            await pilot.pause()

            saved_line = log._lines[0]
            assert saved_line.text.plain == f"Saved: {filename}"
            assert any(
                getattr(span.style, "link", None) == f"file://{url}"
                for span in saved_line.text.spans
            )

            width = log.scrollable_content_region.width
            rows = [
                saved_line.row_text(width, row)
                for row in range(saved_line.row_count(width))
            ]
            linked_text = "".join(
                row.plain[span.start:span.end]
                for row in rows
                for span in row.spans
                if getattr(span.style, "link", None) == f"file://{url}"
            )
            assert len(rows) > 2
            assert linked_text == filename

    run(exercise())


def test_max_lines_trim_keeps_bookkeeping_consistent() -> None:
    """Tail mutations respect the history cap: trimmed lines drop from
    the front, and the row bookkeeping stays in sync with the stored
    lines."""

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(60, 12)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            log.max_lines = 5
            await pilot.pause()

            for i in range(20):
                area.feed([f"line {i}"], "")

            assert doc_lines(log) == [
                "line 16",
                "line 17",
                "line 18",
                "line 19",
                "",
            ]
            assert len(log._row_count) == len(log._lines)
            assert len(log._prefix) == len(log._lines) + 1
            assert log._prefix[0] == 0
            assert log._prefix[-1] == sum(log._row_count)
            assert log.virtual_size.height == sum(log._row_count)

    run(exercise())

def test_wheel_scroll_back_to_end_reasserts_tail_following() -> None:
    """Wheel scrolling back down to the very end re-enables tail
    following, so appended text is still auto-revealed.

    Manual wheel scrolling up detaches from the tail. Scrolling back
    down to the very end must re-attach (like the End key), even when
    the user keeps scrolling once the viewport has already reached the
    bottom: extra notches past the bottom used to leave `follow_tail`
    false, so newly-appended text stayed off screen because the scroll
    position no longer tracked the growing tail.
    """

    async def exercise() -> None:
        app = _HostApp()
        async with app.run_test(size=(60, 10)) as pilot:
            area = app.query_one(WorkerLogContentArea)
            log = area.worker_log
            await pilot.pause()

            def wheel_up() -> None:
                log._on_mouse_scroll_up(
                    events.MouseScrollUp(log, 5, 5, 0, -1, 0, False, False, False)
                )

            def wheel_down() -> None:
                log._on_mouse_scroll_down(
                    events.MouseScrollDown(log, 5, 5, 0, 1, 0, False, False, False)
                )

            # Grow the log well beyond the viewport so it is scrollable.
            for i in range(60):
                area.feed(["line " + str(i)], "")
            await pilot.pause()
            assert log.follow_tail is True
            assert log.scroll_offset.y >= log.max_scroll_y

            # Scroll up with the wheel: detaches from the tail.
            wheel_up()
            await pilot.pause()
            assert log.follow_tail is False
            assert log.scroll_offset.y < log.max_scroll_y

            # Scroll back down with the wheel, including extra notches
            # once the viewport is already at the bottom.
            for _ in range(60):
                wheel_down()
            await pilot.pause()
            assert log.follow_tail is True
            assert log.scroll_offset.y >= log.max_scroll_y

            # Newly appended text must stay auto-revealed (pinned to
            # the tail), not remain off screen at a frozen position.
            area.feed(["line 60"], "")
            await pilot.pause()
            assert log.follow_tail is True
            assert log.scroll_offset.y >= log.max_scroll_y

    run(exercise())
