from __future__ import annotations

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class Qwen3ServerBaseModel(TtsBaseModel):

    INFO = TtsModelType.QWEN3TTS_SERVER.value

    # Note, for the server version, we use the hyperparam constants in Qwen3BaseModel.
    # The SGL-Omni default values are the same as that of the reference implementation.

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        issues = super().get_blocking_issues(project, instance)
        issue = SglOmniUtil.check_readiness(SglOmniUtil.get_base_url())
        if issue:
            issues.append(issue)
        return issues
