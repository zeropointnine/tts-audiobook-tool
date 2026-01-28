from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class OuteBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.OUTE.value

    DEFAULT_TEMPERATURE = 0.4 # from model library code

    def create_speaker(self, path: str) -> dict | str:
        ...


