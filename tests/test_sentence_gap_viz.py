from typing import cast
import tkinter as tk
import tkinter.font as tkfont

import numpy as np
import pytest

from testx.sentence_gap_viz import App, COL_EXTEND, COL_OK, applied_insertion_spans
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.sentence_gap_util import (
    BoundaryGapResult,
    GapStatus,
    LoudnessEnvelope,
    SentenceBoundary,
    SentenceGapParams,
    SentenceGapReport,
    SentenceGapUtil,
)


class FakeCanvas:
    def __init__(self):
        self.rectangles: list[tuple[tuple[float, ...], dict]] = []

    def winfo_width(self):
        return 1000

    def winfo_height(self):
        return 290

    def create_text(self, *args, **kwargs):
        return 1

    def bbox(self, item):
        return (800, 0, 1000, 10)

    def create_line(self, *args, **kwargs):
        pass

    def create_rectangle(self, *args, **kwargs):
        self.rectangles.append((args, kwargs))

    def spans(self, color: str) -> np.ndarray:
        return np.array([(args[0], args[2]) for args, options in self.rectangles
                         if options.get("fill") == color])


def extended(index: int, start: float, end: float, position: float, duration: float) -> BoundaryGapResult:
    return BoundaryGapResult(
        SentenceBoundary(index, start, end), GapStatus.EXTEND,
        gap_start=start, gap_end=end, insert_at=position, insert_seconds=duration,
    )


def make_report(results: list[BoundaryGapResult]) -> SentenceGapReport:
    original = Sound(np.ones(4000, dtype=np.float32), 1000)
    params = SentenceGapParams(fade_ms=0)
    sound, insertions = SentenceGapUtil.apply(original, results, params)
    envelope = LoudnessEnvelope.compute(original, params.frame_ms, params.hop_ms)
    return SentenceGapReport(original, sound, results, insertions, envelope, params)


def make_app(report: SentenceGapReport) -> App:
    app = App.__new__(App)
    app.report = report
    app.loaded = None
    app.selected = -1
    app.small = cast(tkfont.Font, None)
    app.mono = cast(tkfont.Font, None)
    app.peak = 1.0
    app.canvases = {key: cast(tk.Canvas, FakeCanvas()) for key in ("before", "after")}
    return app


def draw(app: App, which: str) -> FakeCanvas:
    assert app.report is not None
    mapper = app.report if which == "after" else None
    sound = app.report.sound if mapper is not None else app.report.original
    # Exercise the real panel caller without creating a Tk window. One pixel is one second.
    app.draw_panel(which, sound, None, (0.0, 1000.0), mapper)
    return cast(FakeCanvas, app.canvases[which])


def test_after_overlays_count_each_insertion_once_and_shift_later_bands():
    # The second orange band must move with the audio, and green ends must not double-count.
    report = make_report([extended(0, .8, 1.2, 1.0, .2), extended(1, 2.8, 3.2, 3.0, .3)])
    canvas = draw(make_app(report), "after")
    np.testing.assert_allclose(canvas.spans(COL_OK), [[.8, 1.4], [3.0, 3.7]])
    np.testing.assert_allclose(canvas.spans(COL_EXTEND), [[1.0, 1.2], [3.2, 3.5]])


def test_before_overlays_keep_original_coordinates():
    # BEFORE shows original gaps and thin insertion ticks, not extended spans.
    report = make_report([extended(0, .8, 1.2, 1.0, .2), extended(1, 2.8, 3.2, 3.0, .3)])
    canvas = draw(make_app(report), "before")
    np.testing.assert_allclose(canvas.spans(COL_OK), [[.8, 1.2], [2.8, 3.2]])
    np.testing.assert_allclose(canvas.spans(COL_EXTEND), [[0.0, 3.0], [2.0, 5.0]])


@pytest.mark.parametrize("position", [1.0004, 1.0006])
def test_after_uses_applied_sample_rounding_for_bands_and_zoom(position: float):
    # Raw positions on either side of sample rounding must not count their own insertion.
    report = make_report([extended(0, .8, 1.2, position, .2006)])
    insertion = report.insertions[0]
    app = make_app(report)
    app.selected = 0
    canvas = draw(app, "after")
    np.testing.assert_allclose(canvas.spans(COL_EXTEND),
                               [[insertion.position, insertion.position + insertion.duration]])
    np.testing.assert_allclose(canvas.spans(COL_OK), [[.8, 1.2 + insertion.duration]])
    assert app.boundary_centers() == pytest.approx(
        (position, insertion.position + insertion.duration / 2))


def test_applied_spans_follow_application_order_and_skip_zero_sample_requests():
    # Results can be out of order; a request rounded to zero has no AFTER band.
    report = make_report([
        extended(0, 2.8, 3.2, 3.0004, .3006),
        extended(1, .8, 1.2, 1.0006, .2006),
        extended(2, 1.8, 2.2, 2.0, .0004),
    ])
    spans = applied_insertion_spans(report)
    assert set(spans) == {0, 1}
    assert spans[1] == pytest.approx((1.001, 1.202))
    assert spans[0] == pytest.approx((3.201, 3.502))
    canvas = draw(make_app(report), "after")
    np.testing.assert_allclose(canvas.spans(COL_EXTEND), [spans[0], spans[1]])


def test_unextended_gap_after_earlier_insertion_is_only_shifted():
    # An already-long gap still follows earlier insertions without growing itself.
    report = make_report([
        extended(0, .8, 1.2, 1.0, .2),
        BoundaryGapResult(SentenceBoundary(1, 2.8, 3.2), GapStatus.LONG_ENOUGH,
                          gap_start=2.8, gap_end=3.2),
    ])
    canvas = draw(make_app(report), "after")
    np.testing.assert_allclose(canvas.spans(COL_OK), [[.8, 1.4], [3.0, 3.4]])
    np.testing.assert_allclose(canvas.spans(COL_EXTEND), [[1.0, 1.2]])
