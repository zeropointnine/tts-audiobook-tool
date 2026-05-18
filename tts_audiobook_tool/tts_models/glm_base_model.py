from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos

class GlmBaseModel(TtsBaseModel):
    
    INFO = TtsModelInfos.GLM.value
    
    SAMPLE_RATES = [24000, 32000]

