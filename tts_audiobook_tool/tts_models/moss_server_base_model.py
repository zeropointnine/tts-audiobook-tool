from __future__ import annotations

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.moss_base_model import MossArchType, MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class MossServerBaseModel(MossBaseModel):
    """ 
    Base model for the 'server' version of MOSS (v1.5)
    """

    INFO = TtsModelType.MOSS_SERVER.value

    # Rem, MOSS can fail to properly terminate gens when hyperparams are out of a certain range,
    # so max tokens should be set as "small" as possble
    MAX_NEW_TOKENS = 1024 

    CONFIG = MossConfigs.DELAY

    def get_loaded_arch_type(self) -> MossArchType:
        """ 
        Must infer architecture type using model id ("good-enough test") 
        """
        model_id = SglOmniUtil.get_model_id()
        if not model_id:
            SglOmniUtil.update_model_id()
            model_id = SglOmniUtil.get_model_id()

        if "local" in model_id.lower():
            return MossArchType.LOCAL

        return MossArchType.DELAY

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        # Overrides with server ping check
        readiness_issue = SglOmniUtil.check_readiness(SglOmniUtil.get_base_url())
        if readiness_issue:
            return [readiness_issue]
        
        return []
