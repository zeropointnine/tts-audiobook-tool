from __future__ import annotations
from abc import abstractmethod
from enum import Enum

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_RECO_RANGE
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.local_definition import require_local_definition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class ChatterboxBaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("chatterbox_local").value

    # Parameter defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("chatterbox_local")

    @classmethod
    def get_max_words_range_reco(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[int, int, str]:
        if project.get_model_setting('chatterbox_local', 'type') in (ChatterboxType.MULTILINGUAL, ChatterboxType.MULTILINGUAL_V2):
            return (40, 40, "Chatterbox Multilingual V2")
        return MAX_WORDS_PER_SEGMENT_RECO_RANGE

    @staticmethod
    def repetition_penalty_setting(model_type: ChatterboxType) -> str:
        """Each variant has its own repetition penalty setting: the values behave differently."""
        if model_type == ChatterboxType.MULTILINGUAL_V2:
            return "ml_v2_repetition_penalty"
        if model_type == ChatterboxType.MULTILINGUAL_V3:
            return "ml_v3_repetition_penalty"
        return "turbo_repetition_penalty"

    @staticmethod
    def default_repetition_penalty(model_type: ChatterboxType) -> float:
        name = ChatterboxBaseModel.repetition_penalty_setting(model_type)
        return float(ChatterboxBaseModel.LOCAL.number(name).default)

    @abstractmethod
    def supported_languages_multi(self) -> list[str]:
        """ List of supported languages (applies to Multilingual variant only) """
        ...

    @classmethod
    def get_blocking_issues(cls, project: Project, instance: TtsBaseModel | None) -> list[ReadinessIssue]:

        if not instance:
            return [] # Can't know if language is invalid w/o loading model code
        else:
            assert(isinstance(instance, ChatterboxBaseModel))
            if not project.language_code in instance.supported_languages_multi():
                return [ReadinessIssue("supported language code", f"Language code {project.language_code} not supported by current model")]
            else:
                return []

    def get_warning_issues(self, project: Project) -> list[str]:
        warnings = super().get_warning_issues(project)
        # Multilingual receives the project code verbatim as `language_id`; Turbo takes none.
        if project.get_model_setting('chatterbox_local', 'type').is_multilingual:
            from tts_audiobook_tool.tts_models.model_support import make_language_value_note
            warnings.append(make_language_value_note(project.language_code, project.language_code))
        return warnings

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        return project.get_model_setting('chatterbox_local', 'type').label # eg, "Chatterbox-Multilingual"

# ---

class ChatterboxType(tuple[str, str, str], Enum):

    # Keep the legacy "multilingual" id mapped to V2 so existing projects do
    # not silently switch checkpoints. V3 is first so new projects default to it.
    MULTILINGUAL_V3 = "multilingual-v3", "Chatterbox-Multilingual V3", "Latest multilingual model"
    MULTILINGUAL_V2 = "multilingual", "Chatterbox-Multilingual V2", "Legacy multilingual model"
    # Source-compatibility alias for callers that used the former member name.
    MULTILINGUAL = "multilingual", "Chatterbox-Multilingual V2", "Legacy multilingual model"
    TURBO = "turbo", "Chatterbox-Turbo", "Distilled, en only"

    @property
    def is_multilingual(self) -> bool:
        return self in (ChatterboxType.MULTILINGUAL_V2, ChatterboxType.MULTILINGUAL_V3)

    @property
    def multilingual_t3_model(self) -> str | None:
        if self == ChatterboxType.MULTILINGUAL_V2:
            return "v2"
        if self == ChatterboxType.MULTILINGUAL_V3:
            return "v3"
        return None

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @property
    def description(self) -> str:
        return self.value[2]

    @staticmethod
    def get_by_id(id: str) -> ChatterboxType | None:
        for item in list(ChatterboxType):
            if id == item.id:
                return item
        return None
