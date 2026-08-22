"""Pixel-diff test: ReflowLog vs RichLog(wrap=True) in Textual 8.2.8.

`ReflowLog`'s reflow is a verbatim replica of rich's ``Text.wrap``
pipeline (``_fast_breaks`` == ``divide_line``, the same ``Text.divide``
row materialization, the same ``crop_extend``/``apply_style`` row
finish), so at any given width its rendered frame must be
cell-for-cell identical to a `RichLog` that wrapped the same content at
that width. This test verifies that end-to-end, headlessly:

  1. WRITE PARITY   - write the same ANSI corpus into a ReflowLog and a
                      fresh RichLog at the same size; full-screen frames
                      must match segment-for-segment (text + style),
                      including at the tail.
  2. REFLOW PARITY  - resize the ReflowLog through a width sequence;
                      after each resize its frame must equal a FRESH
                      RichLog written at the new width with the same
                      content, i.e. reflowing is indistinguishable from
                      re-wrapping at the target width.
  3. ANCHOR         - scrolled away from the tail, the logical line at
                      the top of the viewport stays at the top after a
                      resize (compared against a fresh RichLog scrolled
                      to the same logical line).
  4. CAP / CLEAR    - `max_lines` counts logical lines (not wrapped
                      rows); the visible tail after trimming matches a
                      fresh RichLog holding the same surviving lines,
                      and `clear()` empties the widget like RichLog's.
  5. MOUNT WRITE    - a write made during `on_mount` (before the size is
                      known) lands identically in both widgets.

Corpus note: every batch begins with a short unique marker line and a
>500-cell line, so `RichLog`'s measured max width exceeds the content
region at every test width and its wrap width equals the region width
exactly (with `min_width=1`), matching ReflowLog's wrap width.

Console note: `LogApp` pins `app.console.size` at construction, before
any write, modeling the real TUI's process-startup width snapshot
(`RichLog` measures its content against the console at write time).

Run:  ./venv-base/bin/python -m testx.reflow_log_diff
"""

import asyncio
import random
import sys

from rich.cells import cell_len
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import RichLog

from tts_audiobook_tool.textual.reflow_log import ReflowLog

HEIGHT = 24
BATCH_SIZE = 200
BATCHES = 5
WIDTHS = [100, 80, 120, 40, 200, 58, 60, 100, 150]

WORDS = (
    "the quick brown fox jumps over the lazy dog audiobook tts generation "
    "progress chapter sentence percent chapter token speaker audio"
).split()
CJK = "テキストは高速に折りたたまれ、セル幅が正しく計算されることを確認している。"


def make_batch(batch_no: int) -> list[str]:
    """A batch of ANSI-flavored logical lines with a unique marker and a
    guaranteed-long line (keeps RichLog's measured width above the region
    width at every test width)."""
    rng = random.Random(1000 + batch_no)
    lines = [
        f"##B{batch_no}## " + " ".join(rng.choices(WORDS, k=40)),
        "\x1b[31m" + "x" * 500 + "\x1b[0m",
    ]
    for i in range(2, BATCH_SIZE):
        kind = (batch_no * 1000 + i) % 10
        if kind == 0:
            lines.append(" ".join(rng.choices(WORDS, k=rng.randint(5, 14))))
        elif kind == 1:
            lines.append("".join(f"{j:05d} " for j in range(20)))
        elif kind == 2:
            lines.append("y" * rng.randint(120, 400))
        elif kind == 3:
            lines.append(" ".join(rng.choices(WORDS, k=rng.randint(20, 40))))
        elif kind == 4:
            lines.append(CJK * rng.randint(1, 3))
        elif kind == 5:
            lines.append(
                " ".join(
                    f"{rng.randint(0,999):>4} {rng.randint(0,9999999):>9}  "
                    f"{rng.choice(WORDS):>8}"
                    for _ in range(rng.randint(2, 4))
                )
            )
        elif kind == 6:
            lines.append("   " + " ".join(rng.choices(WORDS, k=rng.randint(4, 30))))
        elif kind == 7:
            lines.append(" ".join(rng.choices(WORDS, k=rng.randint(5, 40))) + "   ")
        elif kind == 8:
            lines.append("")
        else:
            lines.append(
                f"\x1b[32m[{batch_no:02d}:{i:03d}]\x1b[0m "
                + "  ".join(rng.choices(WORDS, k=rng.randint(6, 25)))
            )
    return lines


BATCHES_ALL = [make_batch(i) for i in range(BATCHES)]
ALL_LINES = [line for batch in BATCHES_ALL for line in batch]


class LogApp(App):
    """One log widget filling a fixed-size screen, with identical CSS for
    both widget classes (scrollbar geometry included)."""

    CSS = """
    Screen {
        background: $surface;
        color: $foreground;
    }
    #log {
        height: 1fr;
        width: 100%;
        background: $surface;
        color: $foreground;
        overflow-y: scroll;
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    """

    def __init__(
        self,
        widget: ReflowLog | RichLog,
        width: int,
        write_at_mount: bool = False,
    ) -> None:
        super().__init__()
        # Model the real TUI: the rich console width is a process-startup
        # snapshot, so it must be pinned before any write (including one
        # made during on_mount).
        self.console.size = (width, HEIGHT)
        self.initial_width = width
        self.widget = widget
        self.write_at_mount = write_at_mount

    def compose(self) -> ComposeResult:
        yield self.widget

    def on_mount(self) -> None:
        if self.write_at_mount:
            self.widget.write(Text.from_ansi("\n".join(BATCHES_ALL[0])))


def make_reflow_log(max_lines: int | None = None) -> ReflowLog:
    return ReflowLog(
        id="log",
        max_lines=max_lines,
        markup=False,
        auto_scroll=True,
    )


def make_rich_log() -> RichLog:
    return RichLog(
        id="log",
        max_lines=None,
        min_width=1,
        wrap=True,
        markup=False,
        highlight=False,
        auto_scroll=True,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def boot(app: LogApp, pilot, width: int) -> None:
    """Wait for the first layout after booting at a fixed size.

    The console size was already pinned at construction time; this only
    lets the initial layout pass settle.
    """
    await pilot.pause()


async def write(app: LogApp, pilot, batch: list[str]) -> None:
    app.widget.write(Text.from_ansi("\n".join(batch)))
    await pilot.pause()
    await pilot.pause()  # let the deferred auto-scroll settle


async def write_text(app: LogApp, pilot, text: Text) -> None:
    app.widget.write(text)
    await pilot.pause()
    await pilot.pause()


async def resize(app: LogApp, pilot, width: int) -> None:
    await pilot.resize_terminal(width, HEIGHT)


def frame(app: LogApp):
    """The full screen as a list of (row text, [(text, style), ...])."""
    strips = app.screen._compositor.render_strips()
    return [
        (strip.text, [(seg.text, seg.style) for seg in strip._segments])
        for strip in strips
    ]


def at_tail(log) -> bool:
    return log.max_scroll_y > 0 and log.scroll_offset.y >= log.max_scroll_y


def compare(label: str, a, b) -> int:
    """Compare two full frames; print diffs; return number of differing
    rows."""
    if len(a) != len(b):
        print(f"  [{label}] FAIL (frame height {len(a)} != {len(b)})")
        return 1
    diffs = 0
    for y, ((row_a, segs_a), (row_b, segs_b)) in enumerate(zip(a, b)):
        if row_a != row_b or segs_a != segs_b:
            diffs += 1
            if diffs <= 3:
                print(f"  DIFF row {y}:")
                print(f"    reflow: {row_a!r}")
                print(f"    rich:   {row_b!r}")
                if segs_a != segs_b and len(segs_a) == len(segs_b):
                    for (ta, sa), (tb, sb) in zip(segs_a, segs_b):
                        if ta != tb or sa != sb:
                            print(f"    seg: {(ta, sa)!r} != {(tb, sb)!r}")
                            break
    print(f"  [{label}] {'PASS' if not diffs else 'FAIL'} ({len(a)} rows, {diffs} differ)")
    return diffs


def _clip_segments(segs, width):
    """Clip a row's segments to the first ``width`` cells (drops the
    scrollbar column). Segments are split at widget boundaries by the
    compositor, so none straddles the region edge."""
    out = []
    col = 0
    for text, style in segs:
        w = cell_len(text)
        if col >= width:
            break
        take = min(w, width - col)
        if take > 0:
            out.append((text[:take], style))
        col += w
    return out


def compare_content(label: str, a, b, content_width: int) -> int:
    """Like `compare`, but ignoring the scrollbar column (used when the
    two widgets legitimately hold different total amounts of content,
    so their scrollbar thumbs differ)."""
    diffs = 0
    for y, ((row_a, segs_a), (row_b, segs_b)) in enumerate(zip(a, b)):
        ca = _clip_segments(segs_a, content_width)
        cb = _clip_segments(segs_b, content_width)
        text_a = "".join(t for t, _ in ca)
        text_b = "".join(t for t, _ in cb)
        if text_a != text_b or ca != cb:
            diffs += 1
            if diffs <= 3:
                print(f"  DIFF row {y}:")
                print(f"    reflow: {text_a!r}")
                print(f"    rich:   {text_b!r}")
    print(
        f"  [{label}] {'PASS' if not diffs else 'FAIL'} "
        f"({len(a)} rows x {content_width} cells, {diffs} differ)"
    )
    return diffs


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


async def phase1_write_parity() -> int:
    print("=== 1. write parity (same width, no resize) ===")
    failures = 0
    app_r = LogApp(make_reflow_log(), WIDTHS[0])
    app_s = LogApp(make_rich_log(), WIDTHS[0])
    async with app_r.run_test(size=(WIDTHS[0], HEIGHT)) as pr:
        await boot(app_r, pr, WIDTHS[0])
        async with app_s.run_test(size=(WIDTHS[0], HEIGHT)) as ps:
            await boot(app_s, ps, WIDTHS[0])
            for k in range(BATCHES):
                await write(app_r, pr, BATCHES_ALL[k])
                await write(app_s, ps, BATCHES_ALL[k])
                failures += compare(
                    f"width {WIDTHS[0]}, after batch {k}",
                    frame(app_r),
                    frame(app_s),
                )
                for app in (app_r, app_s):
                    if not at_tail(app.widget):
                        print(
                            f"  [tail after batch {k}] FAIL "
                            f"(scroll_y={app.widget.scroll_offset.y}, "
                            f"max={app.widget.max_scroll_y})"
                        )
                        failures += 1
    return failures


async def phase2_reflow_parity() -> int:
    print("=== 2. reflow parity (resize vs fresh write at target width) ===")
    failures = 0
    app_r = LogApp(make_reflow_log(), WIDTHS[0])
    async with app_r.run_test(size=(WIDTHS[0], HEIGHT)) as pr:
        await boot(app_r, pr, WIDTHS[0])
        for batch in BATCHES_ALL:
            await write(app_r, pr, batch)
        for width in WIDTHS[1:]:
            await resize(app_r, pr, width)
            app_s = LogApp(make_rich_log(), width)
            async with app_s.run_test(size=(width, HEIGHT)) as ps:
                await boot(app_s, ps, width)
                for batch in BATCHES_ALL:
                    await write(app_s, ps, batch)
                failures += compare(
                    f"reflow {WIDTHS[0]}->{width} vs fresh @ {width}",
                    frame(app_r),
                    frame(app_s),
                )
                if not at_tail(app_r.widget):
                    print(f"  [tail after resize to {width}] FAIL")
                    failures += 1
    return failures


async def phase3_anchor() -> int:
    print("=== 3. scroll anchor on resize ===")
    failures = 0
    anchor_width = WIDTHS[1]
    marker = "##B1##"
    idx = BATCH_SIZE  # logical index of batch 1's first line
    app_r = LogApp(make_reflow_log(), WIDTHS[0])
    app_s = LogApp(make_rich_log(), WIDTHS[0])
    async with app_r.run_test(size=(WIDTHS[0], HEIGHT)) as pr:
        await boot(app_r, pr, WIDTHS[0])
        await write(app_r, pr, BATCHES_ALL[0])
        await write(app_r, pr, BATCHES_ALL[1])
        async with app_s.run_test(size=(WIDTHS[0], HEIGHT)) as ps:
            await boot(app_s, ps, WIDTHS[0])
            await write(app_s, ps, BATCHES_ALL[0])
            await write(app_s, ps, BATCHES_ALL[1])
            # Both at the same width, anchored at the same logical line.
            target_r = app_r.widget._prefix[idx]
            target_s = next(
                i
                for i, strip in enumerate(app_s.widget.lines)
                if strip.text.startswith(marker)
            )
            app_r.widget.scroll_to(y=target_r, animate=False)
            app_s.widget.scroll_to(y=target_s, animate=False)
            await pr.pause()
            await ps.pause()
            failures += compare(
                f"anchored at {marker!r}, both @ {WIDTHS[0]}",
                frame(app_r),
                frame(app_s),
            )
            # Resize the reflow log; the anchor must hold, and the window
            # must equal a fresh RichLog at the new width at the same line.
            await resize(app_r, pr, anchor_width)
            app_s2 = LogApp(make_rich_log(), anchor_width)
            async with app_s2.run_test(size=(anchor_width, HEIGHT)) as ps2:
                await boot(app_s2, ps2, anchor_width)
                await write(app_s2, ps2, BATCHES_ALL[0])
                await write(app_s2, ps2, BATCHES_ALL[1])
                target_s2 = next(
                    i
                    for i, strip in enumerate(app_s2.widget.lines)
                    if strip.text.startswith(marker)
                )
                app_s2.widget.scroll_to(y=target_s2, animate=False)
                await ps2.pause()
                failures += compare(
                    f"reflow to {anchor_width} with anchor held",
                    frame(app_r),
                    frame(app_s2),
                )
    return failures


async def phase4_cap_and_clear() -> int:
    print("=== 4. max_lines (logical) and clear ===")
    failures = 0
    cap = 100
    app_r = LogApp(make_reflow_log(max_lines=cap), WIDTHS[0])
    app_s = LogApp(make_rich_log(), WIDTHS[0])
    app_empty = LogApp(make_rich_log(), WIDTHS[0])
    app_fresh = LogApp(make_rich_log(), WIDTHS[0])
    async with app_r.run_test(size=(WIDTHS[0], HEIGHT)) as pr:
        await boot(app_r, pr, WIDTHS[0])
        for batch in BATCHES_ALL:
            await write(app_r, pr, batch)
        log = app_r.widget
        last_plain = Text.from_ansi(ALL_LINES[-1]).plain
        ok_cap = (
            len(log._lines) == cap
            and log._prefix[-1] == log.virtual_size.height
            and log._prefix[-1] > 0
            and log._lines[-1].text.plain == last_plain
        )
        print(
            f"  [cap] {'PASS' if ok_cap else 'FAIL'} (stored={len(log._lines)}, "
            f"rows={log._prefix[-1]}, virtual={log.virtual_size})"
        )
        failures += 0 if ok_cap else 1
        async with app_s.run_test(size=(WIDTHS[0], HEIGHT)) as ps:
            await boot(app_s, ps, WIDTHS[0])
            await write_text(
                app_s, ps, Text.from_ansi("\n".join(ALL_LINES[-cap:]))
            )
            failures += compare(
                "tail window after logical-line cap",
                frame(app_r),
                frame(app_s),
            )
            log.clear()
            await pr.pause()
            async with app_empty.run_test(size=(WIDTHS[0], HEIGHT)) as pe:
                await boot(app_empty, pe, WIDTHS[0])
                failures += compare(
                    "after clear() vs empty RichLog",
                    frame(app_r),
                    frame(app_empty),
                )
                await write(app_r, pr, BATCHES_ALL[0])
                async with app_fresh.run_test(size=(WIDTHS[0], HEIGHT)) as pf:
                    await boot(app_fresh, pf, WIDTHS[0])
                    await write(app_fresh, pf, BATCHES_ALL[0])
                    # The rich oracle keeps all 200 lines (no logical
                    # cap), so only the content region can be compared:
                    # the last 24 physical rows are the same in both, but
                    # the scrollbar thumbs differ (different totals).
                    content_w = app_fresh.widget.scrollable_content_region.width
                    failures += compare_content(
                        "write after clear() vs fresh (content only)",
                        frame(app_r),
                        frame(app_fresh),
                        content_w,
                    )
    return failures


async def phase5_mount_write() -> int:
    print("=== 5. write during on_mount ===")
    failures = 0
    app_r = LogApp(make_reflow_log(), WIDTHS[0], write_at_mount=True)
    app_s = LogApp(make_rich_log(), WIDTHS[0], write_at_mount=True)
    async with app_r.run_test(size=(WIDTHS[0], HEIGHT)) as pr:
        await boot(app_r, pr, WIDTHS[0])
        async with app_s.run_test(size=(WIDTHS[0], HEIGHT)) as ps:
            await boot(app_s, ps, WIDTHS[0])
            failures += compare(
                "on_mount write, at tail", frame(app_r), frame(app_s)
            )
            if not at_tail(app_r.widget):
                print("  [tail after on_mount write] FAIL (reflow)")
                failures += 1
    return failures


async def main() -> None:
    total = 0
    total += await phase1_write_parity()
    total += await phase2_reflow_parity()
    total += await phase3_anchor()
    total += await phase4_cap_and_clear()
    total += await phase5_mount_write()
    print()
    if total:
        print(f"RESULT: FAIL ({total} failing check(s))")
        sys.exit(1)
    print("RESULT: PASS (all frames pixel-identical)")


if __name__ == "__main__":
    asyncio.run(main())
