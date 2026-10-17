"""Shared audiobook direct-voice pan and parallel, continuous room sound.

Channel policy and voice identity belong to callers. One processor lives for the
whole render/playback run; pan may change per call without moving carried tails.
Voice chat continues to use RoomReverbProcessor.apply instead.
"""

from __future__ import annotations

import math

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.room_reverb import RoomReverbProcessor


def center_unity_pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power gains, scaled so centered mono is unity in both channels."""
    if not math.isfinite(pan) or not -1.0 <= pan <= 1.0:
        raise ValueError("Pan must be finite and between -1 and 1")
    if pan == 0:
        return 1.0, 1.0
    theta = (pan + 1) * math.pi / 4
    return math.sqrt(2) * math.cos(theta), math.sqrt(2) * math.sin(theta)


class VoiceSpatialProcessor:
    """Combine panned narration and optional unpanned wet room, then clip once.

    `channels` is the actual output channel count (after device fallback). A mono
    room can feed stereo output without enabling spread. Mono output requires a
    mono room: construct that room after resolving actual playback channels.
    """

    def __init__(self, channels: int, reverb: RoomReverbProcessor | None = None) -> None:
        if channels not in (1, 2):
            raise ValueError("Voice spatial output requires one or two channels")
        if reverb is not None and reverb.channels > channels:
            raise ValueError("Mono output requires a mono room reverb processor")
        self.channels = channels
        self.reverb = reverb

    def drain_tail(self, dtype: np.dtype | type = np.float32) -> Sound | None:
        """Finish a run with its remaining room decay, never panned direct audio.

        No extra frames are produced with reverb off or an already empty room.
        Call only once the final speech/pause has been processed without its
        legacy ``is_last`` fade. This consumes state; ordinary boundaries must
        keep using ``apply`` so tails overlap following narration naturally.
        """
        if self.reverb is None:
            return None
        wet = self.reverb.drain_wet()
        if len(wet) == 0:
            return None
        if self.channels == 2 and wet.ndim == 1:
            wet = np.repeat(wet[:, None], 2, axis=1)
        return Sound(np.clip(wet, -1.0, 1.0).astype(dtype), self.reverb.sr)

    def apply(
            self, sound: Sound, pan: float = 0.0, send_end: int | None = None, is_last: bool = False,
    ) -> Sound:
        """Process mono input without changing frame count, sample rate or dtype.

        Only samples before `send_end` are narration: they receive pan and feed
        reverb. Remaining samples (break effects/pauses) stay dry and centered,
        with the existing room tail beneath them. None means all samples; zero
        advances only the tail. The end fade affects the wet room alone.
        """
        left, right = center_unity_pan_gains(pan)
        data = sound.data
        if data.ndim != 1:
            raise ValueError("Voice spatial processing requires mono input")
        n = len(data)
        narration_end = n if send_end is None else max(0, min(n, send_end))
        dry = data.astype(np.float64)
        if self.channels == 2:
            dry = np.repeat(dry[:, None], 2, axis=1)
            dry[:narration_end] *= (left, right)
        if self.reverb is not None:
            wet = self.reverb.process_wet(sound, narration_end, is_last)
            if self.channels == 2 and wet.ndim == 1:
                wet = wet[:, None]
            dry += wet
        # Preserve headroom through BOTH paths; ffmpeg's integer cast cannot clip.
        return Sound(np.clip(dry, -1.0, 1.0).astype(data.dtype), sound.sr)
