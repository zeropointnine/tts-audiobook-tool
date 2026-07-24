from __future__ import annotations

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class FishS2ServerBaseModel(TtsBaseModel):

    INFO = TtsModelType.FISH_S2_SERVER.value 

    TOP_K_MAX = 30 # Note, this differs from local inference version

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:

        # If has voice file path, must also have transcript
        if ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.FISH_S2_SERVER) and not ProjectVoiceUtil.primary_voice_transcript(project, TtsModelType.FISH_S2_SERVER):
            return [
                ReadinessIssue(
                    "voice clone transcript",
                    "Voice clone transcript required when voice clone path is supplied"
                )
            ]
        
        # Ping server
        readiness_issue = SglOmniUtil.check_readiness(SglOmniUtil.get_base_url())
        if readiness_issue:
            return [readiness_issue]
        
        return []

    def get_warning_issues(self, project: Project) -> list[str]:
        results = []
        results.extend(super().get_warning_issues(project))
        if project.fish_s2_top_k > FishS2ServerBaseModel.TOP_K_MAX:
            s = f"Top_k ({project.fish_s2_top_k}) out of range for server version of Fish S2 Pro inference, "
            s += f"will clamp to {FishS2ServerBaseModel.TOP_K_MAX}"
            results.append(s)
        return results
