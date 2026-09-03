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

    # Defaults keep the shared base concrete enough for type annotations and
    # compatibility, while the registered subclasses below override both.
    INFO = TtsModelType.MOSS_DELAY_SERVER.value
    CONFIG = MossConfigs.DELAY

    # Rem, MOSS can fail to properly terminate gens when hyperparams are out of a certain range,
    # so max tokens should be set as "small" as possble
    MAX_NEW_TOKENS = 1024

    @classmethod
    def get_output_sample_rate(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> int:
        return cls.CONFIG.value.output_sample_rate

    def get_loaded_arch_type(self) -> MossArchType:
        if self.CONFIG == MossConfigs.LOCAL:
            return MossArchType.LOCAL
        return MossArchType.DELAY

    @classmethod
    def can_hallucinate_music(cls, project: Project, instance: TtsBaseModel | None = None) -> bool:
        return cls.CONFIG == MossConfigs.LOCAL

    @classmethod
    def should_trim_trailing_token_noise(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> bool:
        return cls.CONFIG == MossConfigs.LOCAL

    @classmethod
    def get_menu_text(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        return cls.INFO.ui.get("proper_name") or ""

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        # Server generation does not use local rolling continuation, so do not
        # inherit its batch/rolling compatibility blocker.
        readiness_issue = SglOmniUtil.check_readiness(SglOmniUtil.get_base_url())
        return [readiness_issue] if readiness_issue else []
