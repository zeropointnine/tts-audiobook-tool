from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

class GlmBaseModel(TtsBaseModel):
    
    INFO = TtsModelInfos.CHATTERBOX.value
    
    SAMPLE_RATES = [24000, 32000]

