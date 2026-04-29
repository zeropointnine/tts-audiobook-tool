from __future__ import annotations
from abc import abstractmethod
from enum import Enum

from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.constants import COL_DIM
from tts_audiobook_tool.prereqs_util import PrereqError
from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class ChatterboxBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.CHATTERBOX.value

    DEFAULT_EXAGGERATION = 0.5 # ml only, not turbo
    DEFAULT_CFG = 0.5 # ml only, not turbo
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.95 # ml library default is 1.0, turbo library default is 0.95. but anyway.
    DEFAULT_TOP_K = 1000 # turbo only, not ml
    DEFAULT_REPETITION_PENALTY_ML = 2.0
    DEFAULT_REPETITION_PENALTY_TURBO = 1.2

    @abstractmethod
    def supported_languages_multi(self) -> list[str]:
        """ List of supported languages (applies to Multilingual variant only) """
        ...

    @classmethod
    def get_prereq_errors(cls, project: Project, instance: TtsBaseModel | None) -> list[PrereqError]:

        if not instance:
            return [] # Can't know if language is invalid w/o loading model code
        else:
            assert(isinstance(instance, ChatterboxBaseModel))
            if not project.language_code in instance.supported_languages_multi():
                return [PrereqError("supported language code", f"Language code {project.language_code} not supported by current model")]
            else:
                return []

    @classmethod
    def get_strictness_warning(cls, strictness: Strictness, project: Project, instance: TtsBaseModel | None) -> str:
        if strictness >= Strictness.HIGH:
            return "Not recommended with current TTS model"
        return ""

    @classmethod
    def get_model_display_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        return project.chatterbox_type.label # eg, "Chatterbox-Multilingual"

# ---

class ChatterboxType(tuple[str, str, str], Enum):

    MULTILINGUAL = "multilingual", "Chatterbox-Multilingual", "Supports multiple languages"
    TURBO = "turbo", "Chatterbox-Turbo", "Distilled, en only"

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
