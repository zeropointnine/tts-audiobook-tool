"""
Room tone noise bed: one continuous, constant-level noise added under the whole
output (speech, pauses and break effects alike), so the background never drops
to digital silence between and within segments.

Used by concat and realtime playback per the `room_tone` project setting.
TTS output generally has no steady background of its own (its noise comes and
goes with the vocalizations), so the bed is not matched to the audio: it is a
fixed, deliberately neutral background. See docs-dev/room-tone.md.

Levels are noise RMS in dBFS, before loudness normalization. Durations never change.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from tts_audiobook_tool.app_types import Sound


class ContinuousNoise:
    """
    Stateful unit-RMS noise generator: white -> pink filter -> optional band
    limit (high-pass and low-pass). Successive `next()` calls continue the same
    signal, so next(a) ++ next(b) == next(a + b) sample for sample.
    """

    # Paul Kellet's / R. Bristow-Johnson's 4-coefficient pink (1/f) approximation
    PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

    WARMUP_S = 1.0

    def __init__(
            self,
            sr: int,
            seed: int = 0,
            lowpass_hz: float | None = None,
            highpass_hz: float | None = None,
            lowpass_order: int = 4,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._pink_zi = np.zeros(len(ContinuousNoise.PINK_A) - 1)
        sections: list[np.ndarray] = []
        if highpass_hz is not None and 0 < highpass_hz < sr / 2:
            sections.append(np.asarray(signal.butter(2, highpass_hz, btype="high", fs=sr, output="sos")))
        if lowpass_hz is not None and 0 < lowpass_hz < sr / 2:
            sections.append(np.asarray(signal.butter(lowpass_order, lowpass_hz, btype="low", fs=sr, output="sos")))
        self._sos: np.ndarray | None = np.concatenate(sections) if sections else None
        self._sos_zi: np.ndarray | None = np.zeros((self._sos.shape[0], 2)) if self._sos is not None else None
        self._scale = 1.0 / ContinuousNoise._impulse_rms(self._sos)
        self._filter(self._rng.standard_normal(round(ContinuousNoise.WARMUP_S * sr)))

    @staticmethod
    def _impulse_rms(sos: np.ndarray | None) -> float:
        """ Output RMS of the filter chain for unit-variance white input """
        impulse = np.zeros(1 << 16)
        impulse[0] = 1.0
        h = signal.lfilter(ContinuousNoise.PINK_B, ContinuousNoise.PINK_A, impulse)
        if sos is not None:
            h = signal.sosfilt(sos, h)
        return float(np.sqrt(np.sum(np.square(h))))

    def _filter(self, white: np.ndarray) -> np.ndarray:
        y, self._pink_zi = signal.lfilter(
            ContinuousNoise.PINK_B, ContinuousNoise.PINK_A, white, zi=self._pink_zi)
        if self._sos is not None:
            y, self._sos_zi = signal.sosfilt(self._sos, y, zi=self._sos_zi)
        return np.asarray(y)

    def next(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        return (self._filter(self._rng.standard_normal(n)) * self._scale).astype(np.float32)


class NoiseBed:
    """
    One instance per export or realtime run. Feed it the output in order; the
    noise stream continues across calls, fades in at the very start, and fades
    out at the end of the call marked `is_last`.
    """

    # Band limit: TTS speech rarely carries much above ~8 kHz, so noise "air"
    # up there would stand out on good headphones; pink noise's low end reads
    # as rumble. Low-pass is 4th-order Butterworth, high-pass 2nd-order.
    LOWPASS_HZ = 8000.0
    HIGHPASS_HZ = 80.0
    # Fade at the very start and end of the output
    EDGE_FADE_S = 0.05
    # Fixed seed, so re-exports are reproducible
    SEED = 0x5EED

    def __init__(self, sr: int, level_db: float) -> None:
        self.sr = sr
        self.level_db = level_db
        self._gain = 10 ** (level_db / 20)
        self._noise = ContinuousNoise(sr, NoiseBed.SEED, NoiseBed.LOWPASS_HZ, NoiseBed.HIGHPASS_HZ)
        self._fade_n = max(1, round(NoiseBed.EDGE_FADE_S * sr))
        # Samples emitted so far (drives the start-of-output fade-in)
        self._position = 0

    def apply(self, sound: Sound, is_last: bool = False) -> Sound:
        """ Returns `sound` with the bed added (same length) """
        data = sound.data
        n = len(data)
        if data.ndim != 1 or n == 0 or sound.sr != self.sr:
            return sound

        gain = np.full(n, self._gain, dtype=np.float64)
        if self._position < self._fade_n:
            k = min(n, self._fade_n - self._position)
            gain[:k] *= (self._position + np.arange(k)) / self._fade_n
        if is_last:
            k = min(n, self._fade_n)
            gain[n - k:] *= np.linspace(1.0, 0.0, k)
        self._position += n

        out = data + (self._noise.next(n) * gain).astype(data.dtype)
        # Safety net: ffmpeg streaming casts to int16 without clipping
        if np.max(np.abs(out)) > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return Sound(out, sound.sr)
