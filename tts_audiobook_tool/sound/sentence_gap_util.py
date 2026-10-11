"""
Finds the silences at sentence boundaries inside a rendered sound segment and
extends the short ones to a minimum duration.

Why this is not just another use of SilenceUtil.detect_silences():
a sentence pause (~100-300 ms) is indistinguishable, by silence detection alone,
from every other short dip in speech, so the sidecar's word timings are used to
say *where to look*. But Whisper's word timings are imprecise (the span between
one word's end and the next word's start is typically ~0.3 s wide and often
misplaced), so they are treated only as a search window. The audio itself decides
where the gap actually is, and when it cannot be isolated reliably the boundary
is skipped and the audio is left untouched. A skipped boundary is always the safe
outcome; a wrongly-placed insertion (silence in the middle of a word) is not.

Which boundaries count as sentence boundaries is not guessed from the text: each
sidecar timed phrase records the `Reason` the text was split off for (its
`reason`). A phrase with no recorded reason (a sidecar written before reasons were
saved) cannot be classified, so its boundary is skipped.

Pipeline, per boundary:

    word timings -> search window -> loudness envelope valley -> quiet run
    -> sanity checks (depth, width, dominance, position) -> insertion point

The envelope is computed with short frames (10 ms / 2 ms hop by default) so that
short gaps are not eroded the way they are by SilenceUtil's 30 ms frames.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase


@dataclass(frozen=True)
class SentenceGapParams:
    """ Tunables. Defaults are a starting point meant to be refined against real segments. """

    # Only boundaries after a phrase whose Reason is at least this are considered
    # (SENTENCE also admits PARAGRAPH, SPACE_BREAK, SECTION_BREAK).
    min_reason: Reason = Reason.SENTENCE

    # Target: a sentence gap's quiet run is extended until it is this long (seconds)
    min_gap_s: float = 0.4

    # Loudness envelope resolution
    frame_ms: float = 10.0
    hop_ms: float = 2.0

    # How far beyond the Whisper bracket (previous word's end -> next word's
    # start) on each side to look for the real gap.
    search_margin_s: float = 0.2

    # A frame is "quiet" when it is at least this many dB below the clip's
    # speech level, or within floor_tolerance_db of the window's quietest frame,
    # whichever threshold is higher. (SilenceUtil uses 42, which under-measures gaps that have a soft tail.)
    silence_db_below_ref: float = 30.0
    floor_tolerance_db: float = 10.0

    # The window's quietest frame must be at least this far below the speech
    # level, else there is no valley to speak of and the boundary is skipped.
    min_depth_db: float = 30.0

    # A quiet run shorter than this is not a gap (it's a plosive closure, etc).
    min_run_s: float = 0.03

    # Quiet runs separated by less than this are one run (absorbs clicks).
    merge_s: float = 0.012

    # If the runner-up quiet run in the window is at least this fraction of the
    # winner's width, we can't tell which is the sentence gap: skip.
    ambiguity_ratio: float = 0.7

    # Needed extension below this is not worth making
    min_insert_s: float = 0.01

    # Fade applied to the host audio on both sides of an insertion
    fade_ms: float = 4.0

    # Required text similarity (0-1) between a boundary's neighbouring transcript
    # words and the source phrases' end/start words.
    word_match_ratio: float = 0.7


class GapStatus(Enum):
    EXTEND = "extend"  # gap isolated and shorter than the target: will be extended
    LONG_ENOUGH = "long_enough"  # gap isolated and already long enough: untouched
    SKIP = "skip"  # could not isolate reliably: untouched


@dataclass
class SentenceBoundary:
    """ A boundary between two phrases, located in the transcript word timings """

    # Index of the phrase that ends at this boundary
    phrase_index: int
    # Whisper's bracket: end of the last word before, start of the first word after (seconds)
    lo: float
    hi: float
    prev_word: str = ""
    next_word: str = ""
    # The Reason the text was split here (the ending phrase's)
    reason: Reason = Reason.UNDEFINED
    # Non-empty if the boundary could not be located in the transcript; why
    skip_reason: str = ""


@dataclass
class BoundaryGapResult:
    boundary: SentenceBoundary
    status: GapStatus
    # Short machine-readable code for SKIP results, else ""
    reason: str = ""
    # Isolated quiet run, in the coordinates of the analyzed sound
    gap_start: float | None = None
    gap_end: float | None = None
    # Where to insert silence and how much (EXTEND only)
    insert_at: float | None = None
    insert_seconds: float = 0.0
    # Every quiet run overlapping the search window, for diagnostics
    candidates: list[tuple[float, float]] = field(default_factory=list)
    window: tuple[float, float] = (0.0, 0.0)
    ref_db: float = 0.0
    floor_db: float | None = None
    threshold_db: float | None = None

    @property
    def gap_width(self) -> float | None:
        if self.gap_start is None or self.gap_end is None:
            return None
        return self.gap_end - self.gap_start

    def summary(self) -> str:
        b = self.boundary
        head = (f"phrase {b.phrase_index + 1} [{b.reason.json_value}] "
                f"'{b.prev_word.strip()}' | '{b.next_word.strip()}' bracket {b.lo:.2f}-{b.hi:.2f}")
        if self.status is GapStatus.SKIP:
            return f"{head}  SKIP {self.reason}"
        width_ms = (self.gap_width or 0.0) * 1000
        if self.status is GapStatus.LONG_ENOUGH:
            return f"{head}  ok gap {width_ms:.0f}ms"
        return f"{head}  EXTEND gap {width_ms:.0f}ms +{self.insert_seconds * 1000:.0f}ms at {self.insert_at:.3f}"


@dataclass
class SentenceGapInsertion:
    """ Silence inserted at `position` (original sound's time) """

    position: float
    duration: float


@dataclass
class SentenceGapReport:
    original: Sound
    sound: Sound  # after extension (the same object as `original` if nothing was inserted)
    results: list[BoundaryGapResult]
    insertions: list[SentenceGapInsertion]
    envelope: LoudnessEnvelope
    params: SentenceGapParams

    def count(self, status: GapStatus) -> int:
        return sum(1 for r in self.results if r.status is status)


class LoudnessEnvelope:
    """ Short-frame RMS loudness of a sound, in dBFS (digital silence = -140) """

    def __init__(self, db: np.ndarray, hop_s: float, frame_s: float, duration: float):
        self.db = db
        self.hop_s = hop_s
        self.frame_s = frame_s
        self.duration = duration
        # "Speech level": robust against both leading/trailing silence and peaks
        self.ref_db = float(np.percentile(db, 90)) if db.size else -140.0

    @staticmethod
    def compute(sound: Sound, frame_ms: float = 10.0, hop_ms: float = 2.0) -> LoudnessEnvelope:
        x = np.asarray(sound.data)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float64, copy=False)
        n = len(x)
        sr = sound.sr
        hop = max(1, int(round(sr * hop_ms / 1000)))
        half = max(1, int(round(sr * frame_ms / 1000 / 2)))
        centers = np.arange(0, n, hop)
        cumulative = np.concatenate(([0.0], np.cumsum(x * x)))
        a = np.clip(centers - half, 0, n)
        b = np.clip(centers + half, 0, n)
        mean_square = (cumulative[b] - cumulative[a]) / np.maximum(b - a, 1)
        db = 10.0 * np.log10(np.maximum(mean_square, 1e-14))
        return LoudnessEnvelope(db, hop / sr, 2 * half / sr, n / sr)

    def frame_at(self, t: float) -> int:
        return int(min(max(round(t / self.hop_s), 0), max(len(self.db) - 1, 0)))

    def run_times(self, frame_start: int, frame_end: int) -> tuple[float, float]:
        """ Time span of audio known to be quiet for frames [frame_start, frame_end) """
        start = max(0.0, frame_start * self.hop_s - self.frame_s / 2)
        end = min(self.duration, (frame_end - 1) * self.hop_s + self.frame_s / 2)
        return start, end


class SentenceGapUtil:

    DEFAULT_PARAMS = SentenceGapParams()

    # --- Boundaries from the sidecar ---

    @staticmethod
    def find_boundaries(
            timed_phrases: Sequence[TimedPhrase],
            words: Sequence[Mapping[str, Any]],
            params: SentenceGapParams = DEFAULT_PARAMS,
    ) -> list[SentenceBoundary]:
        """
        Locates the boundary after each phrase (except the last) in the transcript words.

        `timed_phrases` and `words` come from the sidecar ("timed_phrases", and the
        "transcript_words" dicts with "start", "end", "word"). Only boundaries whose
        phrase `reason` is at least `params.min_reason` are returned. A phrase with
        no reason is returned too, flagged, since it can't be ruled out.

        A phrase's time_end was derived from the first word of the *next* phrase,
        so the boundary is the transcript word that starts there. A boundary is
        returned with `skip_reason` set, not dropped, when it can't be trusted:
        no recorded reason, or the neighbouring transcript words don't match the
        phrases' end/start words.
        """
        boundaries: list[SentenceBoundary] = []
        starts = np.array([float(w["start"]) for w in words]) if words else np.zeros(0)

        for k in range(len(timed_phrases) - 1):
            phrase = timed_phrases[k]
            reason = phrase.reason
            if reason is not None and reason < params.min_reason:
                continue
            t = phrase.time_end
            boundary = SentenceBoundary(phrase_index=k, lo=t, hi=t, reason=reason or Reason.UNDEFINED)

            if reason is None:
                boundary.skip_reason = "no_reason"
                boundaries.append(boundary)
                continue

            i = int(np.argmin(np.abs(starts - t))) if starts.size else -1
            if i <= 0 or abs(starts[i] - t) > 0.005:
                boundary.skip_reason = "no_word_at_boundary"
                boundaries.append(boundary)
                continue

            prev_word, next_word = words[i - 1], words[i]
            boundary.prev_word = str(prev_word["word"])
            boundary.next_word = str(next_word["word"])
            boundary.lo = min(float(prev_word["end"]), float(next_word["start"]))
            boundary.hi = max(float(prev_word["end"]), float(next_word["start"]))

            last_source = SentenceGapUtil._edge_token(phrase.text, last=True)
            first_source = SentenceGapUtil._edge_token(timed_phrases[k + 1].text, last=False)
            if not (SentenceGapUtil._words_match(last_source, boundary.prev_word, params.word_match_ratio)
                    and SentenceGapUtil._words_match(first_source, boundary.next_word, params.word_match_ratio)):
                boundary.skip_reason = "word_mismatch"
            boundaries.append(boundary)

        return boundaries

    @staticmethod
    def _normalize_token(s: str) -> str:
        return re.sub(r"\W+", "", s.lower())

    @staticmethod
    def _edge_token(text: str, last: bool) -> str:
        """ Last/first whitespace token of `text` that has any letters or digits in it """
        tokens = text.split()
        for token in (reversed(tokens) if last else tokens):
            if SentenceGapUtil._normalize_token(token):
                return token
        return ""

    @staticmethod
    def _words_match(source: str, transcribed: str, min_ratio: float) -> bool:
        a = SentenceGapUtil._normalize_token(source)
        b = SentenceGapUtil._normalize_token(transcribed)
        if not a or not b:
            return False
        return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio

    # --- Analysis ---

    @staticmethod
    def analyze(
            sound: Sound,
            boundaries: Sequence[SentenceBoundary],
            params: SentenceGapParams = DEFAULT_PARAMS,
            envelope: LoudnessEnvelope | None = None,
            words: Sequence[Mapping[str, Any]] = (),
    ) -> list[BoundaryGapResult]:
        """
        Decides whether to extend, leave alone or skip each boundary; does not modify audio.
        When transcript words are supplied, all word boundaries compete for quiet runs,
        including non-sentence pauses in the widened search window.
        """
        env = envelope or LoudnessEnvelope.compute(sound, params.frame_ms, params.hop_ms)
        word_brackets: list[tuple[float, float]] = []
        for previous, following in zip(words, words[1:]):
            end, start = float(previous["end"]), float(following["start"])
            word_brackets.append((min(end, start), max(end, start)))

        results: list[BoundaryGapResult] = []
        for i, boundary in enumerate(boundaries):
            other_brackets = [
                (other.lo, other.hi)
                for j, other in enumerate(boundaries) if j != i and not other.skip_reason
            ]
            competing_words = word_brackets.copy()
            # Exclude this boundary's own word bracket only once. Duplicate
            # brackets remain competitors: their ownership is genuinely ambiguous.
            if (boundary.lo, boundary.hi) in competing_words:
                competing_words.remove((boundary.lo, boundary.hi))
            other_brackets.extend(competing_words)
            results.append(SentenceGapUtil._analyze_one(env, boundary, params, other_brackets))

        # Thresholds vary between windows, so two analyses can measure different
        # extents of the same quiet run. Reject overlapping claims rather than
        # risk extending that physical pause twice, even at different positions.
        conflicts: set[int] = set()
        for i, result in enumerate(results):
            if result.status is GapStatus.SKIP:
                continue
            assert result.gap_start is not None and result.gap_end is not None
            for j in range(i):
                other = results[j]
                if other.status is GapStatus.SKIP:
                    continue
                assert other.gap_start is not None and other.gap_end is not None
                if max(result.gap_start, other.gap_start) < min(result.gap_end, other.gap_end):
                    conflicts.update((i, j))
        for i in conflicts:
            results[i].status = GapStatus.SKIP
            results[i].reason = "shared_gap"
            results[i].insert_at = None
            results[i].insert_seconds = 0.0
        return results

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """ [start, end) index pairs of the True runs in `mask` """
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        d = np.diff(padded)
        return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))

    @staticmethod
    def _merge_runs(runs: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in runs:
            if merged and start - merged[-1][1] <= max_gap_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _analyze_one(
            env: LoudnessEnvelope,
            boundary: SentenceBoundary,
            p: SentenceGapParams,
            other_brackets: Sequence[tuple[float, float]] = (),
    ) -> BoundaryGapResult:

        result = BoundaryGapResult(boundary, GapStatus.SKIP, ref_db=env.ref_db)

        def skip(reason: str) -> BoundaryGapResult:
            result.reason = reason
            return result

        if boundary.skip_reason:
            return skip(boundary.skip_reason)

        # 1. Search window: Whisper's bracket, widened because it is often misplaced
        w0 = max(0.0, boundary.lo - p.search_margin_s)
        w1 = min(env.duration, boundary.hi + p.search_margin_s)
        result.window = (w0, w1)
        f0 = env.frame_at(w0)
        f1 = env.frame_at(w1) + 1
        if f1 - f0 < 3:
            return skip("window_empty")

        # 2. Is there a valley at all?
        floor = float(env.db[f0:f1].min())
        result.floor_db = floor
        if floor > env.ref_db - p.min_depth_db:
            return skip("no_valley")

        # 3. Quiet runs, measured over the whole clip so that a run that
        # continues past the window is measured at its full width
        threshold = max(floor + p.floor_tolerance_db, env.ref_db - p.silence_db_below_ref)
        result.threshold_db = threshold
        merge_frames = int(round(p.merge_s / env.hop_s))
        runs = SentenceGapUtil._merge_runs(SentenceGapUtil._runs(env.db <= threshold), merge_frames)
        in_window = [(fs, fe) for fs, fe in runs if fs < f1 and fe > f0]
        result.candidates = [env.run_times(fs, fe) for fs, fe in in_window]

        widths = [(env.run_times(fs, fe), fs, fe) for fs, fe in in_window]
        widths = [((st, en), fs, fe) for (st, en), fs, fe in widths if en - st >= p.min_run_s]
        if not widths:
            return skip("gap_too_short")

        # Widened windows can include both neighboring sentence pauses and
        # unrelated word pauses. A run belongs only to the closest word-timing
        # bracket, not to every window it overlaps. Ties within one envelope hop
        # are unsafe, so neither boundary claims it.
        def belongs_to_boundary(start: float, end: float) -> bool:
            center = (start + end) / 2
            distance = max(boundary.lo - center, center - boundary.hi, 0.0)
            return all(
                distance + env.hop_s < max(lo - center, center - hi, 0.0)
                for lo, hi in other_brackets
            )

        widths = [item for item in widths if belongs_to_boundary(*item[0])]
        if not widths:
            return skip("gap_owned_by_other_boundary")
        widths.sort(key=lambda item: item[0][1] - item[0][0], reverse=True)

        # 4. The sentence gap should stand out from anything else quiet nearby
        (gap_start, gap_end), fs, fe = widths[0]
        gap_width = gap_end - gap_start
        if len(widths) > 1:
            (s2, e2), _, _ = widths[1]
            if (e2 - s2) >= p.ambiguity_ratio * gap_width:
                return skip("ambiguous")
        if fs == 0 or fe >= len(env.db):
            return skip("gap_at_clip_edge")

        result.gap_start = gap_start
        result.gap_end = gap_end

        need = p.min_gap_s - gap_width
        if need < p.min_insert_s:
            result.status = GapStatus.LONG_ENOUGH
            return result

        # 5. Insert at the middle of the gap's deepest stretch
        sub = env.db[fs:fe]
        core_runs = SentenceGapUtil._runs(sub <= float(sub.min()) + 3.0)
        core_start, core_end = max(core_runs, key=lambda r: r[1] - r[0])
        insert_at = (fs + (core_start + core_end - 1) / 2) * env.hop_s
        if not (w0 <= insert_at <= w1):
            return skip("gap_off_target")

        result.status = GapStatus.EXTEND
        result.insert_at = float(insert_at)
        result.insert_seconds = float(need)
        return result

    # --- Application ---

    @staticmethod
    def apply(
            sound: Sound,
            results: Sequence[BoundaryGapResult],
            params: SentenceGapParams = DEFAULT_PARAMS,
    ) -> tuple[Sound, list[SentenceGapInsertion]]:
        """
        Inserts digital silence at every EXTEND result, fading the host audio
        out and back in around it. Returns the new Sound (the same object if
        there was nothing to do) and the insertions made.
        """
        if sound.data.ndim != 1:
            raise ValueError("mono sound required")

        todo = sorted(
            (r for r in results
             if r.status is GapStatus.EXTEND and r.insert_at is not None and r.insert_seconds > 0),
            key=lambda r: r.insert_at or 0.0,
        )
        if not todo:
            return sound, []

        x = sound.data
        sr = sound.sr
        host: list[np.ndarray] = []
        fill_lengths: list[int] = []
        insertions: list[SentenceGapInsertion] = []
        fade_n = max(0, int(round(sr * params.fade_ms / 1000)))

        previous = 0
        for r in todo:
            assert r.insert_at is not None
            index = min(max(int(round(r.insert_at * sr)), previous), len(x))
            length = int(round(r.insert_seconds * sr))
            if length <= 0:
                continue
            host.append(x[previous:index].copy())
            fill_lengths.append(length)
            insertions.append(SentenceGapInsertion(index / sr, length / sr))
            previous = index
        host.append(x[previous:].copy())

        # Raised-cosine fade of the host out of and back into each silence
        for j, chunk in enumerate(host):
            n = min(fade_n, len(chunk) // 2)
            if n < 2:
                continue
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(x.dtype)
            if j < len(fill_lengths):
                chunk[-n:] *= ramp[::-1]
            if j > 0:
                chunk[:n] *= ramp

        pieces: list[np.ndarray] = []
        for j, chunk in enumerate(host):
            pieces.append(chunk)
            if j < len(fill_lengths):
                pieces.append(np.zeros(fill_lengths[j], dtype=x.dtype))
        return Sound(np.concatenate(pieces), sr), insertions

    @staticmethod
    def map_time(insertions: Sequence[SentenceGapInsertion], t: float) -> float:
        """ Maps a time in the original sound to the same audio's time after insertions """
        return t + sum(i.duration for i in insertions if t > i.position)

    @staticmethod
    def extend_sentence_gaps(
            sound: Sound,
            timed_phrases: Sequence[TimedPhrase],
            words: Sequence[Mapping[str, Any]],
            params: SentenceGapParams = DEFAULT_PARAMS,
    ) -> SentenceGapReport:
        """ Whole pipeline: sidecar timings + audio -> extended audio and a per-boundary report """
        env = LoudnessEnvelope.compute(sound, params.frame_ms, params.hop_ms)
        boundaries = SentenceGapUtil.find_boundaries(timed_phrases, words, params)
        results = SentenceGapUtil.analyze(sound, boundaries, params, env, words=words)
        new_sound, insertions = SentenceGapUtil.apply(sound, results, params)
        return SentenceGapReport(sound, new_sound, results, insertions, env, params)
