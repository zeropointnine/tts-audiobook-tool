from typing import Protocol

from tts_audiobook_tool.tts_model.tts_model import TtsModel

class GlmProtocol(Protocol):

    SAMPLE_RATES = [24000, 32000]

class GlmModelProtocol(TtsModel, GlmProtocol):
    ...
