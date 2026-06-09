from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

class GlmBaseModel(TtsBaseModel):
    
    INFO = TtsModelType.GLM.value
    
    SAMPLE_RATES = [24000, 32000]

