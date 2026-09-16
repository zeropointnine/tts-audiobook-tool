from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class AuKServerBaseModel(TtsBaseModel):
    """Shared non-library functionality for SGL-Omni AuK variants."""

    INFO = TtsModelType.AUK_SERVER.value

    SPEED_DEFAULT = 1.0
    SPEED_MIN = 0.5
    SPEED_MAX = 2.0

    @classmethod
    def get_blocking_issues(
        cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        issues = super().get_blocking_issues(project, instance)
        model_type = TtsModelType.get_by_id(cls.INFO.id)
        references = ProjectVoiceUtil.voice_reference_pairs(project, model_type)

        if references and any(not transcript.strip() for _, transcript in references):
            issues.append(
                ReadinessIssue(
                    "voice clone transcript",
                    "Voice clone transcript required for every AuK voice sample",
                )
            )

        if not any(issue.short == "voice sample" for issue in issues):
            missing_voice = next(
                (
                    voice_file_name
                    for voice_file_name, _ in references
                    if not os.path.exists(
                        ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name)
                    )
                ),
                "",
            )
            if missing_voice:
                issues.append(
                    ReadinessIssue(
                        "voice sample",
                        f"Voice clone sample file not found: {missing_voice}",
                    )
                )

        readiness_issue = SglOmniUtil.check_readiness(SglOmniUtil.get_base_url())
        if readiness_issue:
            issues.append(readiness_issue)

        return issues
