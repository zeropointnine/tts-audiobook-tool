"""
Detection and repair of "hard-gated" audio: stretches of digital silence inside a sound.

WHY THIS EXISTS
    A voice-clone reference whose pauses are digitally silent (a noise gate,
    denoiser or lossy encoder wrote true zeros instead of a room-tone floor) can
    make some TTS models never emit their end-of-audio token. On audio.cpp's
    Higgs Audio v3 that surfaces as HTTP 500 "reached max_tokens ... before EOC"
    after ~14 s, and the app then retries the same line, so one bad reference
    produces a storm of failures. Filling the zero stretches with low-level
    noise (keeping every pause where it was) removes most of those failures.

WHAT IT PROVIDES
    HardGateUtil.inspect(sound)         -> HardGatingInfo (read-only measurement)
    HardGateUtil.fill_core_runs(sound)  -> (Sound, HardGateFillReport) (repair)

STATUS / WHAT IS KNOWN (2026-10-07; full write-up and numbers in
plans/hard-gate-issue.md)
    - Nothing in the app calls this yet. It is a library plus a dev harness
      (testx/reference_gating_ab.py); wiring it into voice import is undecided.
    - Observed only on Higgs v3 (audio.cpp). In A/B tests (same texts and seeds,
      original vs filled reference), Chatterbox, Breeze TTS 2 and Qwen3-TTS had
      zero failures with the original gated references, so they gained nothing.
      Untested: MOSS, Fish, Echo-TTS, OmniVoice, GLM, Higgs v3 via SGL-Omni.
    - The fill is not a complete cure even for Higgs: one reference (Allegra)
      still failed ~1-3 of 40 requests after filling, at every fill level from
      -85 to -55 dBFS, and one line fails with the original reference too.
    - Fill level (default -65 dBFS) made little difference between -85 and -55.
    - A more aggressive variant that also replaced the decaying gate tails
      around each zero run was tried and removed: no better, slightly worse.

THINGS TO KNOW BEFORE CHANGING IT
    - Detection is amplitude-based (|x| <= 1e-5, i.e. -100 dBFS, for >= 50 ms),
      not "== 0", so lossy decodes (tiny non-zero values) are caught too.
    - Repair preserves length, pause count and pause positions, and leaves every
      sample outside the filled runs bit-identical. Keep that invariant: the
      point is to change what the silence contains, not the timing.
    - The noise spectrum is matched to the clip's own quiet frames when there
      are enough of them, otherwise pink. Gated clips often have no ungated
      quiet frames, so pink is common in practice.
    - Gate tails (-100..-70 dBFS) next to a zero run are deliberately left alone.
    - Tests: tests/test_silence_hard_gating.py.
"""

from dataclasses import dataclass

import numpy as np

from tts_audiobook_tool.app_types import Sound


# Hard-gated (digitally silenced) audio: a run of samples at or below
# HARD_GATING_NEAR_ZERO_AMPLITUDE (-100 dBFS) lasting at least
# HARD_GATING_MIN_GAP_MS. Natural recordings keep a room-tone floor (about
# -50 to -90 dBFS), so sustained near-zero runs mean a noise gate, denoiser or
# lossy encoder wrote true silence. A reference sample with such gaps can make
# an in-context (audio prompt) TTS model never emit its end-of-audio token;
# see testx/reference_gating_ab.py for the measurement harness and
# plans/hard-gate-issue.md for the write-up.
HARD_GATING_NEAR_ZERO_AMPLITUDE = 1e-5
HARD_GATING_MIN_GAP_MS = 50.0
HARD_GATING_SEVERE_GAP_MS = 200.0

# Repair defaults
HARD_GATE_FILL_FLOOR_DB = -65.0
HARD_GATE_FADE_MS = 3.0


@dataclass(frozen=True)
class HardGatingInfo:
    """What HardGateUtil.inspect() found in one sound."""

    duration: float
    gap_count: int
    longest_gap_ms: float
    gated_duration: float
    gated_sample_pct: float
    absolute_zero_pct: float
    floor_db: float

    @property
    def is_gated(self) -> bool:
        """Whether any silence run met the gating criteria."""
        return self.gap_count > 0

    @property
    def is_severe(self) -> bool:
        return self.longest_gap_ms >= HARD_GATING_SEVERE_GAP_MS

    def describe(self) -> str:
        """One-line summary suitable for a menu or dialog."""
        if not self.is_gated:
            return f"No hard-gated silence (floor {self.floor_db:.0f} dBFS)"
        return (
            f"Hard-gated silence: {self.gated_sample_pct:.0f}% of samples in "
            f"{self.gap_count} gap(s), longest {self.longest_gap_ms:.0f} ms, "
            f"floor {self.floor_db:.0f} dBFS"
        )


@dataclass(frozen=True)
class HardGateFillReport:
    """What a HardGateUtil.fill_* call did."""

    regions_filled: int
    filled_ms: float
    target_db: float
    level_source: str  # "configured" | "measured" | "none"
    spectrum_source: str  # "matched" | "pink" | "white" | "none"


class HardGateUtil:
    """
    Detection and repair of hard-gated (digitally silent) stretches in a sound,
    chiefly voice-clone reference samples.

    Repair replaces the gated stretches with low-level room-tone noise. Length,
    pause count and pause positions are preserved, and samples outside the
    filled regions are untouched.
    """

    # ---
    # Detection

    @staticmethod
    def inspect(
        sound: Sound,
        near_zero_amplitude: float = HARD_GATING_NEAR_ZERO_AMPLITUDE,
        min_gap_ms: float = HARD_GATING_MIN_GAP_MS,
    ) -> HardGatingInfo:
        """
        Measures hard-gated (digitally silenced) stretches of a sample.

        Criterion is amplitude-based rather than "sample == 0" so lossy sources
        (which decode silence to tiny non-zero values) are caught too. This is
        a read-only measurement: it neither trims nor repairs anything.
        """
        empty = HardGatingInfo(0.0, 0, 0.0, 0.0, 0.0, 0.0, float("-inf"))
        data = sound.data
        if data is None or sound.sr <= 0:
            return empty
        flat = HardGateUtil._to_mono(data)
        if flat.size == 0:
            return empty

        runs = HardGateUtil._find_runs(flat, near_zero_amplitude, min_gap_ms, sound.sr)
        lengths = np.array([end - start for start, end in runs], dtype=np.int64)
        gap_count = len(runs)
        if gap_count:
            longest_ms = 1000.0 * int(lengths.max()) / sound.sr
            gated_samples = int(lengths.sum())
        else:
            longest_ms = 0.0
            gated_samples = 0

        window = max(1, round(0.02 * sound.sr))
        frame_count = flat.size // window
        if frame_count:
            frames = flat[: frame_count * window].reshape(frame_count, window).astype(np.float64)
            frame_db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
            floor_db = float(np.percentile(frame_db, 10))
        else:
            rms = float(np.sqrt((flat.astype(np.float64) ** 2).mean()))
            floor_db = 20.0 * np.log10(rms + 1e-12)

        return HardGatingInfo(
            duration=flat.size / sound.sr,
            gap_count=gap_count,
            longest_gap_ms=longest_ms,
            gated_duration=gated_samples / sound.sr,
            gated_sample_pct=100.0 * gated_samples / flat.size,
            absolute_zero_pct=100.0 * float(np.count_nonzero(flat == 0.0)) / flat.size,
            floor_db=floor_db,
        )

    # ---
    # Repair

    @staticmethod
    def fill_core_runs(
        sound: Sound,
        floor_db: float = HARD_GATE_FILL_FLOOR_DB,
        near_zero_amplitude: float = HARD_GATING_NEAR_ZERO_AMPLITUDE,
        min_gap_ms: float = HARD_GATING_MIN_GAP_MS,
        room_tone: str = "match",
        match_level: bool = False,
        fade_ms: float = HARD_GATE_FADE_MS,
    ) -> tuple[Sound, HardGateFillReport]:
        """
        Fills the detected near-zero runs (>= `min_gap_ms` at or below
        `near_zero_amplitude`) with low-level noise, cross-faded over `fade_ms`
        at each run edge. Samples outside the runs are untouched.

        `room_tone`: "match" (spectrum of the clip's own quiet frames, falling
        back to pink if there are too few), "pink" or "white".
        `match_level`: use the measured room-tone level instead of `floor_db`.
        """
        return HardGateUtil._fill(
            sound, floor_db, near_zero_amplitude, min_gap_ms, room_tone, match_level, fade_ms,
        )

    # ---
    # Internals

    @staticmethod
    def _to_mono(data: np.ndarray) -> np.ndarray:
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32).reshape(-1)

    @staticmethod
    def _find_runs(flat: np.ndarray, near_zero_amplitude: float, min_gap_ms: float, sr: int) -> list[tuple[int, int]]:
        """ (start, end) sample ranges where |x| <= threshold for at least `min_gap_ms`. """
        mask = np.abs(flat) <= near_zero_amplitude
        changes = np.diff(np.concatenate(([0], mask.astype(np.int32), [0])))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        min_samples = max(1, round(min_gap_ms * sr / 1000.0))
        return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_samples]

    @staticmethod
    def _fill(
        sound: Sound,
        floor_db: float,
        near_zero_amplitude: float,
        min_gap_ms: float,
        room_tone: str,
        match_level: bool,
        fade_ms: float,
    ) -> tuple[Sound, HardGateFillReport]:
        sr = sound.sr
        data = HardGateUtil._to_mono(sound.data).copy()
        n_total = data.size

        cores = HardGateUtil._find_runs(data, near_zero_amplitude, min_gap_ms, sr) if sr > 0 else []
        if not cores:
            return Sound(data, sr), HardGateFillReport(0, 0.0, floor_db, "none", "none")

        # Frames overlapping any near-zero sample don't count as natural room tone
        exclude = np.abs(data) <= near_zero_amplitude

        spectrum: tuple[np.ndarray, np.ndarray] | None = None
        spectrum_source = "white" if room_tone == "white" else "pink"
        target_db = floor_db
        level_source = "configured"
        if room_tone == "match":
            estimate = HardGateUtil._estimate_room_tone(data, sr, exclude)
            if estimate is not None:
                measured_spectrum, measured_level, window = estimate
                spectrum = (np.fft.rfftfreq(window, 1.0 / sr), measured_spectrum)
                spectrum_source = "matched"
                if match_level:
                    target_db = 20.0 * np.log10(measured_level + 1e-12)
                    level_source = "measured"
        target_rms = 10.0 ** (target_db / 20.0)

        rng = np.random.default_rng(20261007)
        fade = max(0, round(fade_ms * sr / 1000.0))
        filled = 0
        for start, end in cores:
            n = end - start
            if spectrum is not None:
                src_freqs, src_magnitude = spectrum
                # Interpolated per region because region length varies
                shape = np.interp(np.fft.rfftfreq(n, 1.0 / sr), src_freqs, src_magnitude)
                peak = float(shape.max())
                if peak > 0.0:
                    shape = shape / peak
            elif room_tone == "white":
                shape = np.ones(n // 2 + 1)
            else:
                shape = None  # pink
            noise = HardGateUtil._shaped_noise(n, sr, rng, shape) * target_rms

            # Cross-fade original -> noise at the edges. A side that touches
            # the file boundary has no neighbour to blend with, so stays hard.
            weight = np.ones(n, dtype=np.float32)
            edge = min(fade, n // 2)
            if edge > 0:
                ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge))).astype(np.float32)
                if start > 0:
                    weight[:edge] = ramp
                if end < n_total:
                    weight[-edge:] = ramp[::-1]
            data[start:end] = data[start:end] * (1.0 - weight) + noise * weight
            filled += n

        return Sound(data, sr), HardGateFillReport(
            regions_filled=len(cores),
            filled_ms=1000.0 * filled / sr,
            target_db=target_db,
            level_source=level_source,
            spectrum_source=spectrum_source,
        )

    @staticmethod
    def _pink_shape(freqs: np.ndarray) -> np.ndarray:
        """ 1/sqrt(f) amplitude shape; DC removed. """
        shape = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
        shape[0] = 0.0
        return shape

    @staticmethod
    def _estimate_room_tone(
        data: np.ndarray, sr: int, exclude_mask: np.ndarray
    ) -> tuple[np.ndarray, float, int] | None:
        """ Mean magnitude spectrum, RMS and window size of the clip's own natural quiet frames, or None. """
        window = 1024 if sr >= 32000 else 512
        frame_count = data.size // window
        if frame_count < 8:
            return None
        frames = data[: frame_count * window].reshape(frame_count, window)
        db = 20.0 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-12)
        excluded = exclude_mask[: frame_count * window].reshape(frame_count, window).any(axis=1)
        quiet = (~excluded) & (db < np.percentile(db, 95) - 30.0)
        if int(np.count_nonzero(quiet)) < 5:
            return None
        selected = frames[quiet].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(selected * np.hanning(window), axis=1)).mean(axis=0)
        level = float(np.sqrt((selected ** 2).mean()))
        return spectrum, level, window

    @staticmethod
    def _shaped_noise(n: int, sr: int, rng: np.random.Generator, shape: np.ndarray | None) -> np.ndarray:
        """ Unit-RMS noise of length `n` with the given magnitude shape (None = pink). """
        spectrum = np.fft.rfft(rng.standard_normal(n))
        if shape is None:
            shape = HardGateUtil._pink_shape(np.fft.rfftfreq(n, 1.0 / sr))
        out = np.fft.irfft(spectrum * shape, n=n)
        rms = float(np.sqrt(np.mean(out ** 2)))
        if rms <= 0.0:
            return np.zeros(n, dtype=np.float32)
        return (out / rms).astype(np.float32)
