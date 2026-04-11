from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class MiraBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.MIRA.value

    TEMPERATURE_DEFAULT = 0.7
    TEMPERATURE_MIN = 0.0
    TEMPERATURE_MAX = 2.0
    TOP_P_DEFAULT = 0.95
    TOP_K_DEFAULT = 50
    REPETITION_PENALTY_DEFAULT = 1.2
    MAX_NEW_TOKENS = 2048 # default is 1024, which is enough for ~60 words

    def set_voice_clone(self, path: str) -> None:
        ...

    def clear_voice_clone(self) -> None:
        ...

    def set_params(self, temperature: float, max_new_tokens: int, top_k: int=-1, top_p: float=-1, repetition_penalty: float=-1) -> None:
        ...
