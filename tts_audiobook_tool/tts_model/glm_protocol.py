from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel

class GlmProtocol(Protocol):

    SAMPLE_RATES = [24000, 32000]

    def generate(
        self,
        prompt_text: str,
        prompt_speech: str,
        syn_text: str,
        seed: int
    ) -> Sound | str:
        ...

class GlmModelProtocol(TtsModel, GlmProtocol):
    ...

