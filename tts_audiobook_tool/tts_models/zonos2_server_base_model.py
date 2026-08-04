from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class Zonos2ServerBaseModel(TtsBaseModel):

    INFO = TtsModelType.ZONOS2_SERVER.value

    # Defaults and bounds match the Zonos2 reference UI values.
    TOP_K_DEFAULT = 100
    TOP_K_MIN = 1
    TOP_K_MAX = 200
    TEMPERATURE_DEFAULT = 1.15
    TEMPERATURE_MIN = 0.05
    TEMPERATURE_MAX = 2.0
    REPETITION_PENALTY_DEFAULT = 1.2
    REPETITION_PENALTY_MIN = 1.0
    REPETITION_PENALTY_MAX = 2.0

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        issues = super().get_blocking_issues(project, instance)

        readiness_issue = SglOmniUtil.check_readiness(SglOmniUtil.get_base_url())
        if readiness_issue:
            issues.append(readiness_issue)

        return issues
