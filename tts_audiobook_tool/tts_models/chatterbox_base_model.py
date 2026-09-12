from __future__ import annotations
from abc import abstractmethod
from enum import Enum

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_RECO_RANGE
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class ChatterboxBaseModel(TtsBaseModel):

    INFO = TtsModelType.CHATTERBOX.value

    DEFAULT_EXAGGERATION = 0.5 # ml only, not turbo
    DEFAULT_CFG = 0.5 # ml only, not turbo
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.95 # ml library default is 1.0, turbo library default is 0.95. but anyway.
    DEFAULT_TOP_K = 1000 # turbo only, not ml
    DEFAULT_REPETITION_PENALTY_ML_V2 = 2.0
    DEFAULT_REPETITION_PENALTY_ML_V3 = 1.2
    DEFAULT_REPETITION_PENALTY_ML = DEFAULT_REPETITION_PENALTY_ML_V2
    DEFAULT_REPETITION_PENALTY_TURBO = 1.2

    @classmethod
    def get_max_words_range_reco(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[int, int, str]:
        if project.chatterbox_type in (ChatterboxType.MULTILINGUAL, ChatterboxType.MULTILINGUAL_V2):
            return (40, 40, "Chatterbox Multilingual V2")
        return MAX_WORDS_PER_SEGMENT_RECO_RANGE

    @staticmethod
    def default_repetition_penalty(model_type: ChatterboxType) -> float:
        if model_type == ChatterboxType.MULTILINGUAL_V2:
            return ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_ML_V2
        if model_type == ChatterboxType.MULTILINGUAL_V3:
            return ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_ML_V3
        return ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_TURBO

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

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        return project.chatterbox_type.label # eg, "Chatterbox-Multilingual"

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
