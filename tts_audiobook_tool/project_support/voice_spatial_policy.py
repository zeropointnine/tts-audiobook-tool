"""Run-scoped narration positioning, room reverb, and output-channel policy (no live TTS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import RoomReverb, RoomReverbStereo
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


@dataclass(frozen=True)
class VoiceSpatialPolicy:
    """Snapshot shared by export and realtime, independent of selected range.

    A phrase group's (clamped) voice index identifies the voice used for
    generation, so its stereo position and room reverb follow that voice.
    """

    voice_count: int
    requested_channels: int
    # Per voice slot (`Voice.room_reverb`).
    voice_reverbs: tuple[RoomReverb, ...] = ()
    # Per voice slot (`Voice.pan`).
    voice_pans: tuple[float, ...] = ()
    room_reverb_stereo: RoomReverbStereo = RoomReverbStereo.MONO

    @classmethod
    def from_project(cls, project: Project) -> VoiceSpatialPolicy:
        model_type = project.get_tts_model_type()
        voices = ProjectVoiceUtil.get_voice_values(project, model_type)
        voice_count = len(voices)
        # Independent of the selected model's voice-clone support: existing
        # audio keeps its voices' rooms (e.g. concat with no model selected).
        voice_reverbs = tuple(voice.room_reverb for voice in project.voice_references)
        voice_pans = tuple(voice.pan for voice in project.voice_references)
        stereo_reverb = project.room_reverb_stereo.is_stereo and any(
            reverb is not RoomReverb.OFF for reverb in voice_reverbs
        )
        stereo_positioning = any(pan != 0 for pan in voice_pans)
        return cls(
            voice_count=voice_count,
            requested_channels=2 if stereo_positioning or stereo_reverb else 1,
            voice_reverbs=voice_reverbs,
            room_reverb_stereo=project.room_reverb_stereo,
            voice_pans=voice_pans,
        )

    def pan_for_voice_index(self, index: int) -> float:
        """The stereo position of the voice that generated a phrase group with this `voice_index`."""
        if not self.voice_pans:
            return 0.0
        # Same clamp as generation: -1 (and other stale indices) use the nearest voice
        return self.voice_pans[min(max(index, 0), len(self.voice_pans) - 1)]

    @property
    def stereo_position_summary(self) -> float | None:
        """The one stereo position heard in output, or None when it varies by voice."""
        distinct = set(self.voice_pans)
        if not distinct:
            return 0.0
        return distinct.pop() if len(distinct) == 1 else None

    def room_reverb_for_voice_index(self, index: int) -> RoomReverb:
        """The room reverb of the voice that generated a phrase group with this `voice_index`."""
        if not self.voice_reverbs:
            return RoomReverb.OFF
        # Same clamp as generation: -1 (and other stale indices) use the nearest voice
        return self.voice_reverbs[min(max(index, 0), len(self.voice_reverbs) - 1)]

    @property
    def room_reverb_summary(self) -> RoomReverb | None:
        """The one preset heard in output, or None when it varies by voice."""
        distinct = set(self.voice_reverbs)
        if not distinct:
            return RoomReverb.OFF
        return distinct.pop() if len(distinct) == 1 else None
