from __future__ import annotations

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class PocketBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.POCKET.value

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_INT8 = False
    TEMPERATURE_MIN = 0.01
    TEMPERATURE_MAX = 2.0
    
    # This is pocket default value
    # Higher value may improve sound quality but I didn't hear it, 
    # so leaving it 'un-parameterized'
    LSD = 1 

    # High value here prevents pocket from doing its own internal chunking
    MAX_TOKENS = 120 

    # Predefined voices - stored as bare names, resolved by pocket_tts internals
    # (corresponds to https://huggingface.co/kyutai/pocket-tts/tree/main/embeddings_v3)
    PREDEFINED_VOICES = [
        "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles",
        "cosette", "eponine", "eve", "fantine", "george", "jane", "javert",
        "jean", "marius", "mary", "michael", "paul", "peter_yearsley",
        "stuart_bell", "vera",
    ]

    # See https://huggingface.co/kyutai/pocket-tts/tree/main/languages
    LANGUAGES = [
        "english_2026-04", # same as unqualified "english" 
        "english_2026-01", 
        "french_24l",
        "german_24l",
        "italian",
        "portuguese",
        "spanish_24l",
    ]
    DEFAULT_LANGUAGE = "english_2026-04"

    @classmethod
    def get_voice_value(cls, project: Project) -> str:
        return project.pocket_voice_file_name or project.pocket_predefined_voice

    @classmethod
    def get_model_display_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        s = f"{cls.INFO.ui['proper_name']} {COL_DIM}{project.pocket_model_code}"
        return s
