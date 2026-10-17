"""Run-scoped narration positioning and output-channel policy (no live TTS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


def pan_for_voice_index(index: int) -> float:
    """Positioning is dormant: all voice slots stay centered until configured."""
    return 0.0


@dataclass(frozen=True)
class VoiceSpatialPolicy:
    """Snapshot shared by export and realtime, independent of selected range.

    Only user-defined phrase indices identify the voice used for generation.
    Disabled mode uses voice zero; auto-advance identity cannot be recovered
    from stored phrase metadata and therefore remains centered.
    """

    voice_count: int
    voice_select_mode: VoiceSelectMode
    requested_channels: int

    @classmethod
    def from_project(cls, project: Project) -> VoiceSpatialPolicy:
        voices = ProjectVoiceUtil.get_voice_values(project, project.get_tts_model_type())
        reverb = project.room_reverb
        stereo_reverb = (
            reverb.design is not None
            and reverb.drr_db is not None
            and project.room_reverb_stereo.is_stereo
        )
        # No voice currently has a configured pan, so only active stereo
        # reverb requests stereo. Keep the lookup-based policy for future use.
        stereo_positioning = any(pan_for_voice_index(index) != 0 for index in range(len(voices)))
        return cls(
            voice_count=len(voices),
            voice_select_mode=project.voice_select_mode,
            requested_channels=2 if stereo_positioning or stereo_reverb else 1,
        )

    def pan_for_voice_index(self, index: int) -> float:
        if self.voice_count == 0 or self.voice_select_mode != VoiceSelectMode.USER_DEFINED:
            return 0.0
        # Match GenerateUtil.generate's nearest-valid-index fallback, not the
        # lower-level model's modulo selection. In particular -1 uses voice 0.
        effective_index = min(max(index, 0), self.voice_count - 1)
        return pan_for_voice_index(effective_index)
