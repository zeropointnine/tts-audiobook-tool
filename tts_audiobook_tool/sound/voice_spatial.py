"""Shared audiobook direct-voice pan and parallel, continuous room sound.

Channel policy and voice identity belong to callers. One processor lives for the
whole render/playback run; pan and room may change per call without moving or
cutting off carried tails. Voice chat continues to use RoomReverbProcessor.apply
instead.
"""

from __future__ import annotations

import math

import numpy as np

from tts_audiobook_tool.app_types import RoomReverb, RoomReverbStereo, Sound
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
    """Combine panned narration and unpanned wet rooms, then clip once.

    `channels` is the actual output channel count (after device fallback). A mono
    room can feed stereo output without enabling spread. Mono output requires
    mono rooms: construct the processor after resolving actual playback channels.

    Rooms: each `apply` names the room its narration feeds (`room_reverb`, the
    speaking voice's preset). There is one room per preset, so voices with the
    same preset share a room, made on first use with `room_reverb_stereo`'s
    spread (only with stereo output). Every room advances on every call, so an
    earlier voice's tail keeps decaying under the next voice. A fixed `reverb`
    passed to the constructor is the room fed when `room_reverb` is None.
    """

    def __init__(
            self,
            channels: int,
            reverb: RoomReverbProcessor | None = None,
            *,
            room_reverb_stereo: RoomReverbStereo = RoomReverbStereo.MONO,
    ) -> None:
        if channels not in (1, 2):
            raise ValueError("Voice spatial output requires one or two channels")
        self.channels = channels
        self.room_reverb_stereo = room_reverb_stereo
        # Key None: the fixed room. Insertion order keeps summation deterministic.
        self._rooms: dict[RoomReverb | None, RoomReverbProcessor] = {}
        if reverb is not None:
            self._check_room(reverb)
            self._rooms[None] = reverb

    @property
    def rooms(self) -> dict[RoomReverb | None, RoomReverbProcessor]:
        return dict(self._rooms)

    def _check_room(self, room: RoomReverbProcessor) -> None:
        if room.channels > self.channels:
            raise ValueError("Mono output requires a mono room reverb processor")

    def _room_for(self, room_reverb: RoomReverb | None, sr: int) -> RoomReverbProcessor | None:
        """The room fed by this call's narration, made on first use."""
        if room_reverb is None:
            return self._rooms.get(None)
        if room_reverb.design is None or room_reverb.drr_db is None:
            return None
        room = self._rooms.get(room_reverb)
        if room is None:
            room = RoomReverbProcessor(
                sr, room_reverb.design, room_reverb.drr_db,
                width=self.room_reverb_stereo.width if self.channels == 2 else 0.0,
                spread_early=self.room_reverb_stereo.spread_early,
            )
            self._check_room(room)
            self._rooms[room_reverb] = room
        return room

    def _to_output_shape(self, wet: np.ndarray) -> np.ndarray:
        return wet[:, None] if self.channels == 2 and wet.ndim == 1 else wet

    def drain_tail(self, dtype: np.dtype | type = np.float32) -> Sound | None:
        """Finish a run with its remaining room decay, never panned direct audio.

        Mixes every room's remaining tail. No extra frames are produced with no
        rooms or already empty rooms. Call only once the final speech/pause has
        been processed without its legacy ``is_last`` fade. This consumes state;
        ordinary boundaries must keep using ``apply`` so tails overlap following
        narration naturally.
        """
        tails = [(room.drain_wet(), room.sr) for room in self._rooms.values()]
        tails = [(wet, sr) for wet, sr in tails if len(wet) > 0]
        if not tails:
            return None
        n = max(len(wet) for wet, _ in tails)
        out = np.zeros((n, 2) if self.channels == 2 else n, dtype=np.float64)
        for wet, _ in tails:
            out[:len(wet)] += self._to_output_shape(wet)
        return Sound(np.clip(out, -1.0, 1.0).astype(dtype), tails[0][1])

    def apply(
            self,
            sound: Sound,
            pan: float = 0.0,
            send_end: int | None = None,
            is_last: bool = False,
            room_reverb: RoomReverb | None = None,
    ) -> Sound:
        """Process mono input without changing frame count, sample rate or dtype.

        Only samples before `send_end` are narration: they receive pan and feed
        the room of `room_reverb` (Off: no room; None: the fixed room). Remaining
        samples (break effects/pauses) stay dry and centered, with existing room
        tails beneath them. None means all samples; zero advances only the tails.
        The end fade affects the wet rooms alone.
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
        target = self._room_for(room_reverb, sound.sr)
        for room in self._rooms.values():
            room_send_end = narration_end if room is target else 0
            dry += self._to_output_shape(room.process_wet(sound, room_send_end, is_last))
        # Preserve headroom through BOTH paths; ffmpeg's integer cast cannot clip.
        return Sound(np.clip(dry, -1.0, 1.0).astype(data.dtype), sound.sr)
