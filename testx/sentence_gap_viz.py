"""
Visual + audible test bench for SentenceGapUtil (extending short silences at
sentence boundaries inside a sound segment, using the sidecar's word timings).

    ./venv-base/bin/python -m testx.sentence_gap_viz PATH [--min-gap 0.4]
                                                     [--min-reason {phrase,sentence,paragraph}]

PATH is a sound segment .flac with its .json sidecar beside it (the sidecar needs
`timed_phrases` and `transcript_words`), or a directory of them. Given a file, the
other segments in the same directory can be stepped to as well.

Each sidecar timed phrase records the Reason its text was split off for. Only
boundaries after a phrase whose Reason is at least --min-reason (default: sentence)
are looked at; a phrase with no recorded Reason (an older sidecar) is shown as skipped.
Gaps are extended with digital silence.

Window (top to bottom):
    BEFORE  waveform + Whisper words + boundary overlays, with its loudness envelope (dB) below
    AFTER   same for the sentence-gap-extended audio
    a list of the file's boundaries and what was decided for each, and a detail pane

Overlay colours:
    purple band   Whisper's bracket (previous word's end -> next word's start); dashed = search window
    green band    gap isolated from the audio, long enough already
    orange band   gap isolated and extended (in AFTER, the orange part is the inserted silence)
    red tick      boundary skipped (reason is in the list)
    yellow bars   every quiet run found in the search window (envelope pane, selected boundary only)

Hotkeys: see HELP below, also shown at the bottom of the window.
Playback uses the app's PlaySoundUtil. The playhead is approximate (ignores output latency).
"""

import argparse
import json
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.project_support.sound_segment_util import get_segment_stt_info_path
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.sentence_gap_util import (
    BoundaryGapResult,
    GapStatus,
    SentenceGapParams,
    SentenceGapReport,
    SentenceGapUtil,
)
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

HELP = (
    "1/2 play before/after   Space stop   click panel: play from there   c clip/context playback   "
    "Left/Right boundary   z zoom   a cycle min reason (sentence/phrase/paragraph)   "
    "Tab/Shift+Tab pick param   Up/Down change   0 reset   "
    "n/p next/prev file   x next file with extend   s next file with skip   Esc quit"
)

# Param name, step, min, max (what Tab/Up/Down tunes)
TUNABLE: list[tuple[str, float, float, float]] = [
    ("min_gap_s", 0.05, 0.05, 2.0),
    ("silence_db_below_ref", 2.5, 10.0, 60.0),
    ("min_depth_db", 2.5, 5.0, 80.0),
    ("search_margin_s", 0.05, 0.0, 0.6),
    ("floor_tolerance_db", 1.0, 0.0, 20.0),
    ("ambiguity_ratio", 0.05, 0.3, 1.0),
    ("min_run_s", 0.005, 0.01, 0.15),
    ("merge_s", 0.004, 0.0, 0.05),
    ("fade_ms", 1.0, 0.0, 20.0),
]

ZOOM_HALF_WIDTHS: list[float | None] = [None, 1.5, 0.5]  # None = whole clip
CONTEXT_PLAY_HALF_WIDTH = 1.5

BG = "#1b1d21"
PANEL_BG = "#23262b"
FG = "#d4d7dd"
DIM = "#6b707a"
WAVE = "#6fa8dc"
ENV = "#b9c0cc"
COL_BRACKET = "#8b6bd6"
COL_OK = "#4caf72"
COL_EXTEND = "#ff9f43"
COL_SKIP = "#e5534b"
COL_CAND = "#e6d34f"
COL_WORD = "#9aa3b2"
COL_PLAYHEAD = "#ffffff"

ENV_DB_TOP = 0.0
ENV_DB_BOTTOM = -100.0


@dataclass
class Loaded:
    path: Path
    sound: Sound | None = None
    timed_phrases: list[TimedPhrase] | None = None
    words: list[dict] | None = None
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.sound is not None and self.timed_phrases is not None and self.words is not None


# `a` cycles through these
MIN_REASON_CYCLE = [Reason.SENTENCE, Reason.PHRASE, Reason.PARAGRAPH]


def load_segment(path: Path) -> Loaded:
    """ Loads audio (at the app's rate, as concat would see it) and the sidecar's timings """
    item = Loaded(path)
    loaded = SoundFileUtil.load(str(path), target_sr=APP_SAMPLE_RATE)
    if isinstance(loaded, str):
        item.error = loaded
        return item
    item.sound = loaded

    sidecar = get_segment_stt_info_path(path)
    if not sidecar.exists():
        item.error = f"No sidecar: {sidecar.name}"
        return item
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        item.error = f"Bad sidecar: {e}"
        return item
    if not isinstance(payload, dict) or "timed_phrases" not in payload or not payload.get("transcript_words"):
        item.error = "Sidecar has no timed_phrases/transcript_words (legacy format?)"
        return item
    timed = TimedPhrase.dicts_to_timed_phrases(payload["timed_phrases"])
    if isinstance(timed, str):
        item.error = timed
        return item
    item.timed_phrases = timed
    item.words = payload["transcript_words"]
    return item


def has_usable_sidecar(path: Path) -> bool:
    """ Cheap check used to build the file list: sidecar exists and has >= 2 timed phrases """
    sidecar = get_segment_stt_info_path(path)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return len(payload["timed_phrases"]) >= 2 and bool(payload["transcript_words"])
    except Exception:
        return False


def list_segments(path: Path) -> tuple[list[Path], int]:
    """ Returns navigable files and the starting index """
    if path.is_dir():
        files = sorted(p for p in path.glob("*.flac") if has_usable_sidecar(p))
        return files, 0
    siblings = sorted(p for p in path.parent.glob("*.flac") if p == path or has_usable_sidecar(p))
    return siblings, siblings.index(path)


def applied_insertion_spans(report: SentenceGapReport) -> dict[int, tuple[float, float]]:
    """ AFTER spans by result index, using the sample-rounded insertions actually applied. """
    # Match apply's stable request order, excluding durations that round to zero.
    # Mapping the raw requested position can include its own insertion when the
    # applied position rounded down, so accumulate only preceding insertions.
    requests = sorted(
        ((i, r) for i, r in enumerate(report.results)
         if r.status is GapStatus.EXTEND and r.insert_at is not None
         and r.insert_seconds > 0 and round(r.insert_seconds * report.original.sr) > 0),
        key=lambda item: item[1].insert_at or 0.0,
    )
    spans: dict[int, tuple[float, float]] = {}
    offset = 0.0
    for (i, _), insertion in zip(requests, report.insertions):
        start = insertion.position + offset
        spans[i] = (start, start + insertion.duration)
        offset += insertion.duration
    return spans


class App:

    def __init__(self, files: list[Path], start_index: int, params: SentenceGapParams):
        self.files = files
        self.index = start_index
        self.defaults = params
        self.params = params
        self.param_cursor = 0
        self.zoom_level = 0
        self.context_playback = False
        self.selected = -1

        self.loaded: Loaded | None = None
        self.report: SentenceGapReport | None = None
        self.after_env = None
        self.peak = 1.0

        self.playing: tuple[str, float, float] | None = None  # which, clip offset s, monotonic start
        self.playhead_job: str | None = None

        self.root = tk.Tk()
        self.root.title("sentence gap viz")
        self.root.configure(bg=BG)
        self.root.geometry("1760x1000")
        self.mono = tkfont.Font(family="DejaVu Sans Mono", size=10)
        self.small = tkfont.Font(family="DejaVu Sans Mono", size=8)

        self.header = tk.Label(self.root, text="", anchor="w", justify="left", bg=BG, fg=FG, font=self.mono)
        self.header.pack(fill="x", padx=8, pady=(6, 2))

        self.canvases: dict[str, tk.Canvas] = {}
        for key in ("before", "after"):
            canvas = tk.Canvas(self.root, bg=PANEL_BG, height=290, highlightthickness=0, takefocus=0)
            canvas.pack(fill="x", padx=8, pady=3)
            canvas.bind("<Configure>", lambda e: self.redraw())
            canvas.bind("<Button-1>", lambda e, k=key: self.on_click(k, e.x))
            self.canvases[key] = canvas

        self.listing = tk.Text(self.root, height=9, bg=BG, fg=FG, font=self.mono, bd=0,
                               highlightthickness=0, takefocus=0, wrap="none")
        self.listing.pack(fill="x", padx=8, pady=(4, 0))
        self.listing.tag_configure("cursor", background="#2f3a52")
        self.listing.tag_configure("extend", foreground=COL_EXTEND)
        self.listing.tag_configure("skip", foreground=COL_SKIP)
        self.listing.tag_configure("ok", foreground=COL_OK)
        self.listing.tag_configure("dim", foreground=DIM)

        self.detail = tk.Label(self.root, text="", anchor="w", justify="left", bg=BG, fg=DIM, font=self.mono)
        self.detail.pack(fill="x", padx=8, pady=(2, 0))
        self.status = tk.Label(self.root, text="", anchor="w", bg=BG, fg=COL_CAND, font=self.mono)
        self.status.pack(fill="x", padx=8)
        self.legend = tk.Label(self.root, text=HELP, anchor="w", justify="left", wraplength=1700,
                               bg=BG, fg=DIM, font=self.small)
        self.legend.pack(fill="x", padx=8, pady=(0, 6))

        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    # ---------- data ----------

    def load_current(self) -> None:
        PlaySoundUtil.stop_sound_async()
        self.playing = None
        self.loaded = load_segment(self.files[self.index])
        self.recompute(select_extend=True)

    def recompute(self, select_extend: bool = False) -> None:
        item = self.loaded
        self.report = None
        self.after_env = None
        if item and item.ready:
            assert item.sound and item.timed_phrases and item.words
            self.report = SentenceGapUtil.extend_sentence_gaps(
                item.sound, item.timed_phrases, item.words, self.params)
            from tts_audiobook_tool.sound.sentence_gap_util import LoudnessEnvelope
            self.after_env = LoudnessEnvelope.compute(
                self.report.sound, self.params.frame_ms, self.params.hop_ms)
            peak = max(float(np.max(np.abs(item.sound.data))) if item.sound.data.size else 0.0, 1e-4)
            self.peak = peak
        results = self.report.results if self.report else []
        if select_extend:
            self.selected = next((i for i, r in enumerate(results) if r.status is GapStatus.EXTEND), 0 if results else -1)
        else:
            self.selected = min(self.selected, len(results) - 1) if results else -1
        self.redraw()

    def selected_result(self) -> BoundaryGapResult | None:
        if self.report and 0 <= self.selected < len(self.report.results):
            return self.report.results[self.selected]
        return None

    # ---------- view geometry ----------

    def boundary_centers(self) -> tuple[float, float] | None:
        """ Time of the selected boundary in the before and after sounds """
        r, report = self.selected_result(), self.report
        if r is None or report is None:
            return None
        span = applied_insertion_spans(report).get(self.selected)
        if span is not None and r.insert_at is not None:
            return r.insert_at, (span[0] + span[1]) / 2
        if r.gap_start is not None and r.gap_end is not None:
            c = (r.gap_start + r.gap_end) / 2
        else:
            c = (r.boundary.lo + r.boundary.hi) / 2
        return c, SentenceGapUtil.map_time(report.insertions, c)

    def views(self) -> tuple[tuple[float, float], tuple[float, float]]:
        item, report = self.loaded, self.report
        before_dur = item.sound.duration if item and item.sound else 1.0
        after_dur = report.sound.duration if report else before_dur
        half = ZOOM_HALF_WIDTHS[self.zoom_level]
        centers = self.boundary_centers()
        if half is None or centers is None:
            full = (0.0, max(before_dur, after_dur))
            return full, full
        return (centers[0] - half, centers[0] + half), (centers[1] - half, centers[1] + half)

    # ---------- drawing ----------

    def redraw(self) -> None:
        self.update_texts()
        for canvas in self.canvases.values():
            canvas.delete("all")
        item, report = self.loaded, self.report
        if item is None or item.sound is None:
            return
        (b0, b1), (a0, a1) = self.views()
        self.draw_panel("before", item.sound, report.envelope if report else None, (b0, b1), mapper=None)
        if report is not None and self.after_env is not None:
            self.draw_panel("after", report.sound, self.after_env, (a0, a1), mapper=report)
        self.draw_playhead()

    def draw_panel(self, which: str, sound: Sound, env, view: tuple[float, float], mapper: SentenceGapReport | None) -> None:
        c = self.canvases[which]
        width = max(c.winfo_width(), 100)
        height = max(c.winfo_height(), 100)
        v0, v1 = view
        pps = width / (v1 - v0)

        def tx(t: float) -> float:
            return (t - v0) * pps

        strip_y = 18  # top of the three overlay strips
        wave_y0, wave_y1 = 18 + 24, int(height * 0.60)
        env_y0, env_y1 = wave_y1 + 22, height - 18
        mid = (wave_y0 + wave_y1) / 2

        c.create_text(6, 2, anchor="nw", fill=FG, font=self.mono,
                      text=f"{which.upper()}   {sound.duration:.3f}s   {sound.sr} Hz")

        # --- overlays under the waveform ---
        self.draw_overlays(c, which, tx, strip_y, wave_y0, env_y1, mapper)

        # --- waveform ---
        data = sound.data
        n = len(data)
        sr = sound.sr
        c.create_line(0, mid, width, mid, fill="#343942")
        idx = np.floor(v0 * sr + np.arange(width + 1) * ((v1 - v0) * sr / width)).astype(np.int64)
        half_h = (wave_y1 - wave_y0) / 2
        for x in range(width):
            a, b = int(idx[x]), max(int(idx[x + 1]), int(idx[x]) + 1)
            if b <= 0 or a >= n:
                continue
            seg = data[max(a, 0):min(b, n)]
            lo, hi = float(seg.min()), float(seg.max())
            c.create_line(x, mid - hi / self.peak * half_h, x, mid - lo / self.peak * half_h + 1, fill=WAVE)

        # --- words ---
        item = self.loaded
        if item and item.words:
            for k, w in enumerate(item.words):
                s, e = float(w["start"]), float(w["end"])
                if mapper is not None:
                    s, e = SentenceGapUtil.map_time(mapper.insertions, s), SentenceGapUtil.map_time(mapper.insertions, e)
                if e < v0 or s > v1:
                    continue
                x = tx(s)
                y = wave_y0 + 2 + (k % 2) * 11
                c.create_line(x, wave_y0, x, wave_y1, fill="#3a3f4a", dash=(2, 4))
                c.create_text(x + 2, y, anchor="nw", fill=COL_WORD, font=self.small, text=str(w["word"]).strip()[:14])

        # --- time axis ---
        span = v1 - v0
        step = next((s for s in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0) if s * pps >= 70), 5.0)
        t = np.ceil(v0 / step) * step
        while t <= v1:
            x = tx(float(t))
            c.create_line(x, wave_y1, x, wave_y1 + 4, fill=DIM)
            c.create_text(x + 2, wave_y1 + 4, anchor="nw", fill=DIM, font=self.small, text=f"{t:.2f}")
            t += step

        # --- envelope ---
        if env is not None:
            self.draw_envelope(c, env, view, width, env_y0, env_y1, which)

    def draw_overlays(self, c: tk.Canvas, which: str, tx, strip_y: int, y0: int, y1: int,
                      mapper: SentenceGapReport | None) -> None:
        """
        Solid strips above the waveform (row 1 Whisper bracket, row 2 isolated gap,
        row 3 insertion) plus thin full-height edge lines, so nothing hides the waveform.
        """
        report = self.report
        if report is None:
            return
        ins = report.insertions
        spans = applied_insertion_spans(report) if mapper is not None else {}

        def m(t: float) -> float:
            return SentenceGapUtil.map_time(ins, t) if mapper is not None else t

        rows = (strip_y, strip_y + 7, strip_y + 14)
        h = 5
        labels = (("whisper bracket", COL_BRACKET), ("isolated gap", COL_OK), ("insertion", COL_EXTEND))
        # Legend, laid out right-to-left along the panel's title row
        lx = self.canvases[which].winfo_width() - 6
        for text, color in reversed(labels):
            item = c.create_text(lx, 2, anchor="ne", fill=color, font=self.small, text=f"■ {text}")
            lx = c.bbox(item)[0] - 14

        for i, r in enumerate(report.results):
            selected = (i == self.selected)
            b = r.boundary
            lw = 2 if selected else 1

            if which == "before" or r.status is GapStatus.SKIP:
                # Whisper's bracket is drawn on BEFORE (its own coordinates). On AFTER it is
                # drawn only for skipped boundaries, to show where the skipped boundary was.
                c.create_rectangle(tx(m(b.lo)), rows[0], max(tx(m(b.hi)), tx(m(b.lo)) + 2), rows[0] + h,
                                   fill=COL_BRACKET, width=0)
            if selected and which == "before" and r.status is not GapStatus.SKIP:
                for edge in (r.window[0], r.window[1]):
                    c.create_line(tx(edge), y0, tx(edge), y1, fill=COL_BRACKET, dash=(4, 3))

            if r.status is GapStatus.SKIP:
                x = tx(m((b.lo + b.hi) / 2))
                c.create_line(x, rows[1], x, y1, fill=COL_SKIP, width=lw)
                c.create_rectangle(x - 3, rows[1], x + 3, rows[1] + h, fill=COL_SKIP, width=0)
                continue

            assert r.gap_start is not None and r.gap_end is not None
            # The mapped end already includes any silence inserted inside this gap.
            g0, g1 = m(r.gap_start), m(r.gap_end)
            c.create_rectangle(tx(g0), rows[1], tx(g1), rows[1] + h, fill=COL_OK, width=0)
            for edge in (g0, g1):
                c.create_line(tx(edge), rows[1], tx(edge), y1, fill="#2f6b45", width=lw)

            if r.status is GapStatus.EXTEND and r.insert_at is not None:
                if mapper is None:
                    x = tx(r.insert_at)
                    c.create_rectangle(x - 1, rows[2], x + 2, rows[2] + h, fill=COL_EXTEND, width=0)
                    c.create_line(x, rows[2], x, y1, fill=COL_EXTEND, width=lw)
                elif i in spans:
                    a, z = (tx(t) for t in spans[i])
                    c.create_rectangle(a, rows[2], z, rows[2] + h, fill=COL_EXTEND, width=0)
                    for edge in (a, z):
                        c.create_line(edge, rows[2], edge, y1, fill=COL_EXTEND, width=lw)

    def draw_envelope(self, c: tk.Canvas, env, view: tuple[float, float], width: int, y0: int, y1: int, which: str) -> None:
        v0, v1 = view
        c.create_rectangle(0, y0, width, y1, outline="#343942")

        def ydb(db: float) -> float:
            db = min(max(db, ENV_DB_BOTTOM), ENV_DB_TOP)
            return y0 + (ENV_DB_TOP - db) / (ENV_DB_TOP - ENV_DB_BOTTOM) * (y1 - y0)

        for db in (-20, -40, -60, -80):
            c.create_line(0, ydb(db), width, ydb(db), fill="#2d3138")
            c.create_text(width - 4, ydb(db) - 1, anchor="se", fill=DIM, font=self.small, text=f"{db} dB")
        c.create_line(0, ydb(env.ref_db), width, ydb(env.ref_db), fill="#7d8aa0", dash=(6, 4))
        c.create_text(4, ydb(env.ref_db) + 1, anchor="nw", fill="#7d8aa0", font=self.small, text="speech level")

        r = self.selected_result()
        if which == "before" and r is not None and r.threshold_db is not None:
            c.create_line(0, ydb(r.threshold_db), width, ydb(r.threshold_db), fill=COL_CAND, dash=(3, 3))
            c.create_text(4, ydb(r.threshold_db) + 1, anchor="nw", fill=COL_CAND, font=self.small,
                          text=f"quiet threshold {r.threshold_db:.0f} dB")
            for s, e in r.candidates:
                xs, xe = (s - v0) / (v1 - v0) * width, (e - v0) / (v1 - v0) * width
                c.create_rectangle(xs, y1 - 7, xe, y1 - 1, fill=COL_CAND, width=0)

        points: list[float] = []
        hop = env.hop_s
        for x in range(width):
            f0 = int((v0 + x / width * (v1 - v0)) / hop)
            f1 = max(int((v0 + (x + 1) / width * (v1 - v0)) / hop), f0 + 1)
            if f1 <= 0 or f0 >= len(env.db):
                if points:
                    c.create_line(*points, fill=ENV)
                    points = []
                continue
            points += [x, ydb(float(env.db[max(f0, 0):min(f1, len(env.db))].min()))]
        if len(points) >= 4:
            c.create_line(*points, fill=ENV)

    def draw_playhead(self) -> None:
        for canvas in self.canvases.values():
            canvas.delete("playhead")
        if self.playing is None:
            return
        which, offset, started = self.playing
        t = offset + (time.monotonic() - started)
        view = self.views()[0 if which == "before" else 1]
        canvas = self.canvases[which]
        x = (t - view[0]) / (view[1] - view[0]) * max(canvas.winfo_width(), 100)
        canvas.create_line(x, 0, x, canvas.winfo_height(), fill=COL_PLAYHEAD, width=2, tags="playhead")

    def update_texts(self) -> None:
        item, report = self.loaded, self.report
        name = self.files[self.index].name
        pname, pstep, _, _ = TUNABLE[self.param_cursor]
        half = ZOOM_HALF_WIDTHS[self.zoom_level]
        params = "  ".join(
            (f"[{n}={getattr(self.params, n):g}]" if k == self.param_cursor else f"{n}={getattr(self.params, n):g}")
            for k, (n, _, _, _) in enumerate(TUNABLE))
        head = (f"{self.index + 1}/{len(self.files)}  {name}\n"
                f"view: {'full clip' if half is None else f'zoom +-{half}s'}   "
                f"playback: {'context +-%.1fs' % CONTEXT_PLAY_HALF_WIDTH if self.context_playback else 'whole clip'}   "
                f"boundaries with reason >= {self.params.min_reason.json_value} ({self.params.min_reason.name.lower()})\n{params}")
        self.header.config(text=head)

        self.listing.config(state="normal")
        self.listing.delete("1.0", "end")
        if item and item.error:
            self.listing.insert("end", item.error + "\n", "skip")
        elif report is not None:
            ext = report.count(GapStatus.EXTEND)
            added = sum(i.duration for i in report.insertions)
            self.listing.insert(
                "end",
                f"{len(report.results)} boundaries: {ext} extend, {report.count(GapStatus.LONG_ENOUGH)} long enough, "
                f"{report.count(GapStatus.SKIP)} skip   +{added * 1000:.0f}ms\n", "dim")
            for i, r in enumerate(report.results):
                tag = {"extend": "extend", "long_enough": "ok", "skip": "skip"}[r.status.value]
                line_start = self.listing.index("end-1c")
                self.listing.insert("end", f"{'>' if i == self.selected else ' '} {i + 1:2d}. {r.summary()}\n", tag)
                if i == self.selected:
                    self.listing.tag_add("cursor", line_start, f"{line_start} lineend+1c")
            if self.selected >= 0:
                self.listing.see(f"{self.selected + 2}.0")
        self.listing.config(state="disabled")

        r = self.selected_result()
        if r is None:
            self.detail.config(text="")
        else:
            cand = ", ".join(f"{s:.3f}-{e:.3f}" for s, e in r.candidates) or "none"
            floor = f"{r.floor_db:.0f}" if r.floor_db is not None else "-"
            depth = f"{r.ref_db - r.floor_db:.0f}" if r.floor_db is not None else "-"
            self.detail.config(text=(
                f"selected #{self.selected + 1}: {r.status.value} {r.reason}   window {r.window[0]:.3f}-{r.window[1]:.3f}   "
                f"speech {r.ref_db:.0f} dB  window floor {floor} dB  depth {depth} dB\n"
                f"  quiet runs in window: {cand}"))

    # ---------- playback ----------

    def sound_for(self, which: str) -> Sound | None:
        item, report = self.loaded, self.report
        if item is None or item.sound is None:
            return None
        return item.sound if which == "before" or report is None else report.sound

    def play(self, which: str, from_time: float = 0.0, force_context: bool = False) -> None:
        sound = self.sound_for(which)
        if sound is None:
            return
        start, end = max(from_time, 0.0), sound.duration
        centers = self.boundary_centers()
        if (self.context_playback or force_context) and centers is not None and from_time == 0.0:
            c = centers[0 if which == "before" else 1]
            start, end = max(c - CONTEXT_PLAY_HALF_WIDTH, 0.0), min(c + CONTEXT_PLAY_HALF_WIDTH, sound.duration)
        if end - start < 0.01:
            return
        clip = Sound(np.ascontiguousarray(sound.data[int(start * sound.sr):int(end * sound.sr)]), sound.sr)
        PlaySoundUtil.play_sound_async(clip)
        self.playing = (which, start, time.monotonic())
        self.schedule_playhead()

    def stop(self) -> None:
        PlaySoundUtil.stop_sound_async()
        self.playing = None
        self.draw_playhead()

    def schedule_playhead(self) -> None:
        if self.playhead_job is None:
            self.playhead_job = self.root.after(30, self.tick_playhead)

    def tick_playhead(self) -> None:
        self.playhead_job = None
        if self.playing is None:
            return
        if not PlaySoundUtil.current_sound_id() and time.monotonic() - self.playing[2] > 0.3:
            self.playing = None
            self.draw_playhead()
            return
        self.draw_playhead()
        self.schedule_playhead()

    def on_click(self, which: str, x: int) -> None:
        canvas = self.canvases[which]
        view = self.views()[0 if which == "before" else 1]
        t = view[0] + x / max(canvas.winfo_width(), 100) * (view[1] - view[0])
        sound = self.sound_for(which)
        if sound is not None and 0.0 <= t < sound.duration:
            self.play(which, from_time=t)

    # ---------- navigation ----------

    def set_status(self, text: str) -> None:
        self.status.config(text=text)
        self.root.update_idletasks()

    def go(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.files)
        self.set_status("")
        self.load_current()

    def jump_to_file_with(self, status: GapStatus, direction: int = 1) -> None:
        """ Scans onward (wrapping) for the next file where the current params give >= 1 result of `status` """
        total = len(self.files)
        for step in range(1, total + 1):
            k = (self.index + direction * step) % total
            self.set_status(f"scanning {step}/{total} ...")
            item = load_segment(self.files[k])
            if not item.ready:
                continue
            assert item.sound and item.timed_phrases and item.words
            report = SentenceGapUtil.extend_sentence_gaps(
                item.sound, item.timed_phrases, item.words, self.params)
            if report.count(status) > 0:
                self.index = k
                self.set_status(f"found {status.value} in file {k + 1}")
                self.loaded = item
                PlaySoundUtil.stop_sound_async()
                self.playing = None
                self.recompute()
                self.selected = next(i for i, r in enumerate(report.results) if r.status is status)
                self.redraw()
                return
        self.set_status(f"no file has a '{status.value}' result with these params")

    def on_key(self, event: tk.Event) -> str:
        key = event.keysym
        results = self.report.results if self.report else []

        if key in ("Escape", "q"):
            self.quit()
        elif key == "1":
            self.play("before")
        elif key == "2":
            self.play("after")
        elif key == "space":
            self.stop()
        elif key == "c":
            self.context_playback = not self.context_playback
            self.redraw()
        elif key == "z":
            self.zoom_level = (self.zoom_level + 1) % len(ZOOM_HALF_WIDTHS)
            self.redraw()
        elif key == "a":
            k = MIN_REASON_CYCLE.index(self.params.min_reason) if self.params.min_reason in MIN_REASON_CYCLE else -1
            self.params = replace(self.params, min_reason=MIN_REASON_CYCLE[(k + 1) % len(MIN_REASON_CYCLE)])
            self.recompute(select_extend=True)
        elif key in ("Left", "Right") and results:
            self.selected = (self.selected + (1 if key == "Right" else -1)) % len(results)
            self.redraw()
        elif key in ("Tab", "ISO_Left_Tab"):
            self.param_cursor = (self.param_cursor + (-1 if key == "ISO_Left_Tab" else 1)) % len(TUNABLE)
            self.redraw()
        elif key in ("Up", "Down"):
            name, step, lo, hi = TUNABLE[self.param_cursor]
            value = round(min(max(getattr(self.params, name) + (step if key == "Up" else -step), lo), hi), 4)
            self.params = replace(self.params, **{name: value})
            self.recompute()
        elif key == "0":
            self.params = self.defaults
            self.recompute()
        elif key in ("n", "Next"):
            self.go(1)
        elif key in ("p", "Prior"):
            self.go(-1)
        elif key == "x":
            self.jump_to_file_with(GapStatus.EXTEND)
        elif key == "s":
            self.jump_to_file_with(GapStatus.SKIP)
        return "break"

    def quit(self) -> None:
        PlaySoundUtil.stop_sound_async()
        self.root.destroy()

    def run(self) -> None:
        self.root.update()
        self.load_current()
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="sound segment .flac (with .json sidecar), or a directory of them")
    parser.add_argument("--min-gap", type=float, default=SentenceGapParams().min_gap_s,
                        help="target minimum gap in seconds (default: %(default)s)")
    parser.add_argument("--min-reason", choices=["phrase", "sentence", "paragraph"], default="sentence",
                        help="only consider boundaries whose phrase Reason is at least this (default: %(default)s)")
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    if not path.exists():
        sys.exit(f"Not found: {path}")
    files, start = list_segments(path)
    if not files:
        sys.exit(f"No .flac files with a usable sidecar (>= 2 timed phrases + transcript_words) in {path}")

    params = replace(SentenceGapParams(), min_gap_s=args.min_gap,
                     min_reason=Reason[args.min_reason.upper()])
    App(files, start, params).run()


if __name__ == "__main__":
    main()
