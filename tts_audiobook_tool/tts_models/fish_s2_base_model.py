from enum import Enum

from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos


class FishS2BaseModel(TtsBaseModel):

    INFO = TtsModelInfos.FISH_S2.value

    DEFAULT_COMPILE_ENABLED = True

    DEFAULT_TEMPERATURE = 0.7 # from gradio demo
    MIN_TEMPERATURE = 0.01
    MAX_TEMPERATURE = 1.99 # library asserts temperature < 2
    DEFAULT_TOP_K = 30 # library function default value
    DEFAULT_TOP_P = 0.9 # library function default value

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...

# ---

class FishS2VoiceCloneMode(tuple[str, str, str], Enum):
    CLONE = (
        "clone",
        "Default",
        "Use reference audio as a speaker/timbre prompt."
    )
    ROLLING_CONTINUATION = (
        "rolling_continuation",
        "Rolling Continuation (experimental)",
        "Within a paragraph, uses generated history as context for each next segment"
    )

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
    def get_default() -> "FishS2VoiceCloneMode":
        return FishS2VoiceCloneMode.CLONE

    @staticmethod
    def get_by_id(id: str) -> "FishS2VoiceCloneMode | None":
        for item in list(FishS2VoiceCloneMode):
            if id == item.id:
                return item
        return None

