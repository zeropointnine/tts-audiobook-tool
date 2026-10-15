"""
Room reverb: convolution with a synthetic room impulse response (IR), streamed
over the whole output so tails decay naturally across segment boundaries and
pauses instead of resetting with each segment.

Used by concat, realtime playback and voice chat per the `room_reverb`
project setting.
Only narration feeds the reverb; appended break sound effects pass through dry
(though the tail decays underneath them). Applied before the room tone noise
bed, so the bed isn't reverberated. See docs-dev/room-reverb.md.

The output is always the same length as the input (the tail past the end of
the output is dropped), so durations and timing maps are unchanged.

Optionally stereo (per `room_reverb_stereo`): the voice stays mono
and centered, and the reverb is spread across the channels (see
`make_stereo_irs`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
from scipy import signal

from tts_audiobook_tool.app_types import Sound


@dataclass(frozen=True)
class RoomIrParams:
    """ Synthetic room IR design (see `make_room_ir`) """
    rt60: float = 0.3                # Late tail decay time to -60 dB (s)
    predelay_ms: float = 10.0        # Direct sound -> first reflection
    er_count: int = 10               # Early reflections
    er_window_ms: float = 30.0       # Spread of early reflections after the pre-delay
    er_burst_ms: float = 1.5         # Each reflection is a decaying noise burst this long
    late_db: float = -6.0            # Late tail energy relative to early reflection energy
    late_onset_ms: float = 5.0       # Tail starts fading in this long after the pre-delay...
    late_ramp_ms: float = 25.0       # ...over this long
    hf_split_hz: float = 3000.0      # Above this, the tail decays faster
    hf_rt60_factor: float = 0.6
    highpass_hz: float = 200.0       # Reverb EQ: keeps the room out of the low end
    lowpass_hz: float = 7000.0       # ...and out of the "air"
    seed: int = 0x0EEB


# Room designs, by the key `RoomReverb` presets refer to. A design must never
# change once shipped (saved project settings name it); add a new key instead.
#
# Small: short, early-reflection-weighted. Only works at subtle levels: energy
# in the first ~50 ms fuses with the direct sound and is heard as timbre, and in
# mono pushed harder it turns comb-like ("phasey").
# Medium: bigger room, longer and more diffuse tail, darker and thinner reverb
# EQ; more audible space at much less coloration.
ROOM_DESIGNS: dict[str, RoomIrParams] = {
    "small": RoomIrParams(),
}
ROOM_DESIGNS["medium"] = replace(
    ROOM_DESIGNS["small"],
    rt60=0.8, predelay_ms=15.0, er_count=24, er_window_ms=60.0, er_burst_ms=3.0,
    late_db=3.0, late_onset_ms=10.0, late_ramp_ms=40.0, highpass_hz=250.0, lowpass_hz=6000.0,
)


def _butter_sos(order: int, hz: float, btype: str, sr: int) -> np.ndarray:
    return np.asarray(signal.butter(order, hz, btype=btype, fs=sr, output="sos"))


def make_room_ir(p: RoomIrParams, sr: int) -> np.ndarray:
    """
    Unit-energy synthetic room IR (reverb only, no direct sound): pre-delay,
    early reflections, then an exponentially decaying late tail whose upper
    band decays faster. Deterministic for given params and sample rate.
    """
    rng = np.random.default_rng(p.seed)
    predelay = p.predelay_ms / 1000
    length_s = predelay + p.rt60 * 1.05 + 0.02
    n = round(length_s * sr)
    ir_early = np.zeros(n)

    # Early reflections: jittered evenly through the window, decaying at the
    # tail's rate, random polarity. Smearing each into a short burst avoids
    # discrete taps, which in mono read as hollow comb filtering.
    burst_n = max(2, round(p.er_burst_ms / 1000 * sr))
    burst_env = np.exp(-np.arange(burst_n) / (burst_n / 3))
    for k in range(p.er_count):
        t = predelay + (p.er_window_ms / 1000) * (k + rng.uniform(0, 1)) / p.er_count
        if k == 0:
            t = predelay
        amp = np.exp(-6.91 * (t - predelay) / p.rt60) * rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
        burst = rng.standard_normal(burst_n) * burst_env
        burst /= np.sqrt(np.sum(burst ** 2))
        i = round(t * sr)
        ir_early[i:i + burst_n] += amp * burst[:max(0, n - i)]
    ir_early /= np.sqrt(np.sum(ir_early ** 2))

    # Late tail: exponentially decaying noise with a raised-cosine onset; the
    # upper band decays faster, as in a real (absorptive) room.
    t = np.arange(n) / sr - predelay
    onset = np.clip((t - p.late_onset_ms / 1000) / (p.late_ramp_ms / 1000), 0, 1)
    onset = 0.5 - 0.5 * np.cos(np.pi * onset)
    noise = rng.standard_normal(n)
    low = signal.sosfilt(_butter_sos(4, p.hf_split_hz, "low", sr), noise)
    high = noise - low
    t_pos = np.maximum(t, 0)
    ir_late = (
        low * np.exp(-6.91 * t_pos / p.rt60)
        + high * np.exp(-6.91 * t_pos / (p.rt60 * p.hf_rt60_factor))
    ) * onset
    ir_late *= np.sqrt(10 ** (p.late_db / 10) / np.sum(ir_late ** 2))

    return finish_ir(ir_early + ir_late, p, sr)


def finish_ir(ir: np.ndarray, p: RoomIrParams, sr: int) -> np.ndarray:
    """ Reverb EQ (band limit), end fade, unit energy """
    ir = np.concatenate([ir, np.zeros(round(0.02 * sr))])  # Room for the filters to ring out
    ir = np.asarray(signal.sosfilt(_butter_sos(2, p.highpass_hz, "high", sr), ir))
    ir = np.asarray(signal.sosfilt(_butter_sos(4, p.lowpass_hz, "low", sr), ir))
    fade_n = round(0.01 * sr)
    ir[-fade_n:] *= np.linspace(1, 0, fade_n)
    return ir / np.sqrt(np.sum(ir ** 2))


@cache
def _design_ir(design: str, sr: int) -> np.ndarray:
    ir = make_room_ir(ROOM_DESIGNS[design], sr)
    ir.setflags(write=False)
    return ir


# Early/late split for stereo spread: complementary raised-cosine crossfade
# this long, centered at the end of the early reflection window
SPLIT_XFADE_MS = 6.0


def split_ir(ir: np.ndarray, p: RoomIrParams, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """ (early, late) parts of an IR; they sum to the IR """
    t = np.arange(len(ir)) / sr * 1000 - (p.predelay_ms + p.er_window_ms)
    w = np.clip(t / SPLIT_XFADE_MS + 0.5, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    return ir * (1 - w), ir * w


def make_stereo_irs(
        p: RoomIrParams, sr: int, width: float, spread_early: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    (left, right) IRs: the room's mono IR, with its late tail (and, with
    `spread_early`, its early reflections) spread across the channels. Per
    spread part, mid/side:

        L = c*A + s*B        R = c*A - s*B        c = cos(t), s = sin(t), t = width * pi/4

    A is the mono IR's part, B the same part of a second rendering of the room
    with a different seed (uncorrelated, energy-matched). So each channel gets
    the mono IR's reverb energy (to within the seeds' randomness), interchannel
    correlation of the part is c^2 - s^2 (0 at width 1), and summed to mono B
    cancels: the fold-down is the mono reverb with the spread part lowered by
    -20*log10(c) dB (3 dB at width 1), never wetter or differently colored.
    """
    a_early, a_late = split_ir(make_room_ir(p, sr), p, sr)
    b_early, b_late = split_ir(make_room_ir(replace(p, seed=p.seed + 1), sr), p, sr)
    theta = width * np.pi / 4
    c, s = np.cos(theta), np.sin(theta)

    def spread(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = b * np.sqrt(np.sum(a ** 2) / np.sum(b ** 2))
        return c * a + s * b, c * a - s * b

    late_l, late_r = spread(a_late, b_late)
    early_l, early_r = spread(a_early, b_early) if spread_early else (a_early, a_early)
    return early_l + late_l, early_r + late_r


@cache
def _design_stereo_irs(design: str, sr: int, width: float, spread_early: bool) -> tuple[np.ndarray, np.ndarray]:
    irs = make_stereo_irs(ROOM_DESIGNS[design], sr, width, spread_early)
    for ir in irs:
        ir.setflags(write=False)
    return irs


class StreamingReverb:
    """
    Dry + convolution reverb, fed the output in order. The reverb tail carries
    across calls, so process(a) ++ process(b) == process(a ++ b), and the output
    is the same length as the input.

    `ir_unit` is a unit-energy reverb-only IR; `drr_db` is the
    direct-to-reverberant ratio (dry energy over reverb energy, dB), so higher
    is drier.
    """

    def __init__(self, ir_unit: np.ndarray, drr_db: float) -> None:
        self._ir = ir_unit * 10 ** (-drr_db / 20)
        self._tail = np.zeros(len(self._ir) - 1)

    def process(self, data: np.ndarray, send_end: int | None = None) -> np.ndarray:
        """
        Returns `data` plus reverb (float64). Only `data[:send_end]` feeds the
        reverb (all of it when None); the rest passes through dry, with any
        earlier tail still decaying underneath.
        """
        n = len(data)
        if n == 0:
            return np.asarray(data, dtype=np.float64)
        send = np.asarray(data, dtype=np.float64)
        if send_end is not None and send_end < n:
            send = send.copy()
            send[max(0, send_end):] = 0.0
        wet = signal.oaconvolve(send, self._ir)
        wet[:len(self._tail)] += self._tail
        self._tail = wet[n:].copy()
        return np.asarray(data, dtype=np.float64) + wet[:n]


class RoomReverbProcessor:
    """
    One instance per export or realtime run, for one `RoomReverb` preset's room
    design and level, and optionally a stereo spread (`RoomReverbStereo`'s
    width and spread_early). Feed it the mono output in order (see `apply`).
    Mono in, mono out, unless stereo: then (n, 2) out, the dry signal
    identical in both channels.
    """

    # With `is_last`, the reverb (not the dry signal) fades out over this long
    # at the end of the call, so a truncated tail doesn't stop abruptly
    END_FADE_S = 0.05

    def __init__(
            self, sr: int, design: str, drr_db: float, width: float = 0.0, spread_early: bool = False
    ) -> None:
        self.sr = sr
        if width > 0:
            ir_l, ir_r = _design_stereo_irs(design, sr, width, spread_early)
            self._reverbs = [StreamingReverb(ir_l, drr_db), StreamingReverb(ir_r, drr_db)]
        else:
            self._reverbs = [StreamingReverb(_design_ir(design, sr), drr_db)]

    @property
    def channels(self) -> int:
        return len(self._reverbs)

    def apply(self, sound: Sound, send_end: int | None = None, is_last: bool = False) -> Sound:
        """
        Returns `sound` with reverb added (same length; stereo if the processor is).
        `send_end`: only samples before this index feed the reverb (eg, the
        narration, but not an appended break sound effect).
        """
        data = sound.data
        n = len(data)
        if data.ndim != 1 or n == 0 or sound.sr != self.sr:
            return sound

        outs = [reverb.process(data, send_end) for reverb in self._reverbs]
        out = outs[0] if len(outs) == 1 else np.stack(outs, axis=1)
        if is_last:
            k = min(n, round(RoomReverbProcessor.END_FADE_S * self.sr))
            dry_end = data[n - k:].astype(np.float64)
            fade = np.linspace(1.0, 0.0, k)
            if out.ndim == 2:
                dry_end, fade = dry_end[:, None], fade[:, None]
            out[n - k:] = dry_end + (out[n - k:] - dry_end) * fade
        # Safety net: ffmpeg streaming casts to int16 without clipping
        if np.max(np.abs(out)) > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return Sound(out.astype(data.dtype), sound.sr)
