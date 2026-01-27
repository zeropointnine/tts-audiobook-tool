from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class HiggsProtocol(Protocol):

    # from higgs project README example code
    # (nb, higgs library code uses a default of 1.0 which is much too high for narration)
    DEFAULT_TEMPERATURE = 0.3

    def generate(
            self,
            p_voice_path: str,
            p_voice_transcript: str,
            text: str,
            seed: int,
            temperature: float
    ) -> Sound | str:
        ...

class HiggsModelProtocol(TtsModel, HiggsProtocol):
    ...


