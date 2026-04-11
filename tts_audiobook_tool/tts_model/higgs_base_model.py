from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class HiggsBaseModel(TtsBaseModel):
    
    INFO = TtsModelInfos.HIGGS.value

    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.95


