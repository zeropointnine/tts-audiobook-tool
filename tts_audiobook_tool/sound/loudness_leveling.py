"""
Per-segment loudness leveling: each sound segment gets one gain that brings its
integrated loudness (EBU R128 / ITU-R BS.1770) to the project's loudness target,
followed by a lookahead peak limiter that enforces the target's peak ceiling.

Replaces the earlier whole-file, two-pass ffmpeg loudnorm. Saved segments are
peak normalized, which is a poor proxy for perceived loudness, so adjacent
segments (especially of different voice clones) could differ by several LU; a
whole-file pass can't fix differences between segments. See
docs/loudness-leveling.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d

from tts_audiobook_tool.app_types import NormalizationSpecs, Sound
from tts_audiobook_tool.sound.loudness_lufs_util import LoudnessLufsUtil
from tts_audiobook_tool.sound.voice_spatial import center_unity_pan_gains


@dataclass(frozen=True)
class LevelingResult:
    sound: Sound
    # None when the segment was too short or silent to measure
    measured_lufs: float | None
    gain_db: float
    # Largest gain reduction applied by the limiter (0 when it didn't engage)
    limiter_reduction_db: float
    # Whether the gain hit MAX_BOOST_DB / MAX_CUT_DB
    clamped: bool = False


@dataclass
class LevelingStats:
    """ Running totals over a render, for the post-render report """
    count: int = 0
    unmeasured: int = 0
    clamped: int = 0
    limited: int = 0
    max_limiter_reduction_db: float = 0.0
    # Final safety limiter on mixed output (see LoudnessLeveling.limit_mix)
    mix_limited: int = 0
    max_mix_reduction_db: float = 0.0

    def add(self, result: LevelingResult) -> None:
        self.count += 1
        if result.measured_lufs is None:
            self.unmeasured += 1
        if result.clamped:
            self.clamped += 1
        if result.limiter_reduction_db > 0:
            self.limited += 1
            self.max_limiter_reduction_db = max(self.max_limiter_reduction_db, result.limiter_reduction_db)

    def add_mix(self, reduction_db: float) -> None:
        if reduction_db > 0:
            self.mix_limited += 1
            self.max_mix_reduction_db = max(self.max_mix_reduction_db, reduction_db)

    def summary(self, specs: NormalizationSpecs) -> str:
        s = f"Loudness leveling ({specs.label}, {specs.i:g} LUFS): {self.count} segments"
        details = []
        if self.limited:
            details.append(f"peak-limited {self.limited} (max {self.max_limiter_reduction_db:.1f} dB)")
        if self.mix_limited:
            details.append(f"mix-limited {self.mix_limited} (max {self.max_mix_reduction_db:.2f} dB)")
        if self.clamped:
            details.append(f"gain clamped {self.clamped}")
        if self.unmeasured:
            details.append(f"too short to measure {self.unmeasured}")
        return s + (", " + ", ".join(details) if details else "")


class LoudnessLeveling:

    # The leveling gain never moves a segment by more than this, so a whispered
    # or shouted line (or a glitch) isn't dragged all the way to the target
    MAX_BOOST_DB = 6.0
    MAX_CUT_DB = 10.0

    # pyloudnorm's integrated loudness needs at least one 400 ms gating block.
    # Measuring short segments directly proved better than inferring their gain
    # from neighbours.
    MIN_MEASURE_SECONDS = 0.4

    # The limiter works on sample peaks; this margin covers inter-sample peaks,
    # the little extra that room reverb adds, and AAC encoding overshoot
    TRUE_PEAK_MARGIN_DB = 0.5

    # Typical integrated loudness of a peak-normalized (-1 dBFS) speech segment.
    # Levels that were authored against un-leveled speech (room tone presets,
    # break effect assets) are shifted by (target - this) so they keep their
    # relation to the speech. Measured median on a 4-voice project: -15.7.
    REFERENCE_SPEECH_LUFS = -16.0

    # Limiter: gain reductions are fully in place before a peak arrives
    # (lookahead), then recover at a fixed rate
    LIMITER_LOOKAHEAD_S = 0.005
    LIMITER_RELEASE_DB_PER_S = 30.0

    @staticmethod
    def target_lufs(specs: NormalizationSpecs, channels: int = 1) -> float:
        """
        Integrated loudness target for a mono segment. BS.1770 sums the
        channels' energy, so mono audio played identically on both channels of
        a stereo output measures 10*log10(2) = 3.01 dB louder; the mono segment
        is aimed that much lower so the stereo output measures on target. The
        center-unity pan law keeps a panned voice's total energy the same as a
        centered one's, so this holds for panned voices too.
        """
        return specs.i - 10 * math.log10(max(1, channels))

    @staticmethod
    def reference_offset_db(specs: NormalizationSpecs, channels: int = 1) -> float:
        """
        Gain for audio whose level is defined relative to un-leveled speech
        (room tone, break effects), and the fallback gain for segments too short
        or silent to measure (ie, "assume typical loudness").
        """
        return LoudnessLeveling.target_lufs(specs, channels) - LoudnessLeveling.REFERENCE_SPEECH_LUFS

    @staticmethod
    def pan_boost_db(pan: float) -> float:
        """ How far panning raises the louder channel (center-unity pan law) """
        louder = max(center_unity_pan_gains(pan))
        return max(0.0, 20 * math.log10(louder)) if louder > 0 else 0.0

    @staticmethod
    def ceiling_db_for(specs: NormalizationSpecs, pan: float = 0.0) -> float:
        """
        Sample-peak ceiling for a mono segment before panning, such that the
        panned output stays under the target's peak limit.
        """
        return specs.tp - LoudnessLeveling.TRUE_PEAK_MARGIN_DB - LoudnessLeveling.pan_boost_db(pan)

    @staticmethod
    def measure_lufs(sound: Sound) -> float | None:
        """ Integrated loudness, or None if too short or silent to measure """
        if sound.data.size == 0 or sound.duration < LoudnessLeveling.MIN_MEASURE_SECONDS:
            return None
        lufs = LoudnessLufsUtil.calculate_integrated_loudness(sound.data, sound.sr)
        if lufs is None or not math.isfinite(lufs):
            return None
        return float(lufs)

    @staticmethod
    def gain_db_for(measured_lufs: float | None, specs: NormalizationSpecs, channels: int = 1) -> float:
        """ Leveling gain toward the target, clamped. Unmeasurable: typical-loudness gain """
        if measured_lufs is None:
            return LoudnessLeveling.reference_offset_db(specs, channels)
        gain = LoudnessLeveling.target_lufs(specs, channels) - measured_lufs
        return float(min(max(gain, -LoudnessLeveling.MAX_CUT_DB), LoudnessLeveling.MAX_BOOST_DB))

    @staticmethod
    def level(
            sound: Sound, specs: NormalizationSpecs, pan: float = 0.0, channels: int = 1
    ) -> LevelingResult:
        """
        Levels one segment: measure, apply the clamped gain, then limit peaks to
        the target's ceiling (allowing for the pan boost to come).
        `channels`: of the eventual output (see `target_lufs`).
        Mono or (n, ch) input; returns the same shape and dtype.
        """
        if sound.data.size == 0:
            return LevelingResult(sound, None, 0.0, 0.0)
        measured = LoudnessLeveling.measure_lufs(sound)
        gain_db = LoudnessLeveling.gain_db_for(measured, specs, channels)
        clamped = measured is not None and gain_db != LoudnessLeveling.target_lufs(specs, channels) - measured
        data = sound.data.astype(np.float64) * (10 ** (gain_db / 20))
        data, reduction_db = LoudnessLeveling.limit(data, sound.sr, LoudnessLeveling.ceiling_db_for(specs, pan))
        return LevelingResult(
            Sound(data.astype(sound.data.dtype), sound.sr), measured, gain_db, reduction_db, clamped
        )

    @staticmethod
    def limit_mix(sound: Sound, specs: NormalizationSpecs) -> tuple[Sound, float]:
        """
        Final safety limiter on mixed output (after room reverb and room tone),
        which can push peaks a little past the per-segment ceiling: room reverb
        adds its wet signal on top of the leveled narration. Normally a no-op.
        Returns (sound, largest gain reduction in dB).
        """
        data, reduction_db = LoudnessLeveling.limit(sound.data, sound.sr, LoudnessLeveling.ceiling_db_for(specs))
        if reduction_db == 0.0:
            return sound, 0.0
        return Sound(data.astype(sound.data.dtype), sound.sr), reduction_db

    @staticmethod
    def limit(data: np.ndarray, sr: int, ceiling_db: float) -> tuple[np.ndarray, float]:
        """
        Lookahead peak limiter (channel-linked for (n, ch) input).
        Returns (limited data, largest gain reduction in dB).

        1. Required gain per sample: g = min(1, ceiling / |x|).
        2. Forward-looking minimum over the lookahead window, so the reduction
           is reached before the peak.
        3. Release: the reduction (in dB) decays at a fixed rate after the peak.
        4. Backward moving average over the same window, which smooths the
           attack into a ramp. Every value averaged at a peak's sample was
           already at or below that peak's required gain, so the ceiling holds.
        """
        if data.size == 0:
            return data, 0.0
        ceiling = 10 ** (ceiling_db / 20)
        envelope = np.abs(data) if data.ndim == 1 else np.max(np.abs(data), axis=1)
        if float(np.max(envelope)) <= ceiling:
            return data, 0.0

        n = len(envelope)
        window = max(1, round(LoudnessLeveling.LIMITER_LOOKAHEAD_S * sr))
        required = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-12))

        # [2] min over [i, i + window - 1]; past the end needs no reduction
        padded = np.concatenate([required, np.ones(window)])
        centered = minimum_filter1d(padded, size=window, mode="nearest")
        forward_min = centered[window // 2: window // 2 + n]

        # [3] reduction_out[i] = max over k <= i of (reduction[k] - rate * (i - k))
        reduction = -20 * np.log10(forward_min)
        rate = LoudnessLeveling.LIMITER_RELEASE_DB_PER_S / sr
        ramp = rate * np.arange(n)
        reduction = np.maximum.accumulate(reduction + ramp) - ramp
        gain = 10 ** (-reduction / 20)

        # [4] mean over [i - window + 1, i], padded at the start with gain[0]
        # (itself a minimum over the first window, so the guarantee holds)
        padded = np.concatenate([np.full(window - 1, gain[0]), gain])
        cumulative = np.concatenate([[0.0], np.cumsum(padded)])
        smoothed = (cumulative[window:] - cumulative[:-window]) / window

        out = data * (smoothed if data.ndim == 1 else smoothed[:, None])
        # Float rounding only
        out = np.clip(out, -ceiling, ceiling)
        return out, float(-20 * np.log10(np.min(smoothed)))
