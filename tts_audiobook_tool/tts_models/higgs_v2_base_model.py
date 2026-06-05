from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos


class HiggsV2BaseModel(TtsBaseModel):
    
    INFO = TtsModelInfos.HIGGS_V2.value

    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.95


