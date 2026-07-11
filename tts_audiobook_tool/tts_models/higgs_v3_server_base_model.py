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


class HiggsV3ServerBaseModel(TtsBaseModel):

    INFO = TtsModelType.HIGGS_V3_SERVER.value
    DEFAULT_TEMPERATURE = 1.0
    MAX_TEMPERATURE = 2.0
    DEFAULT_TOP_P = 1.0
    DEFAULT_TOP_K = 100
    MAX_TOKENS = 1536 # 1024 is probably enough for 80 words but

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:

        # If has voice file path, must also have transcript
        if ProjectVoiceUtil.primary_voice_value(project, "higgs_v3_voice_target") and not ProjectVoiceUtil.primary_voice_transcript(project, "higgs_v3_voice_transcript"):
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
