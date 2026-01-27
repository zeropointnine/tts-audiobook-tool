from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class IndexTts2Protocol(Protocol):

    DEFAULT_EMO_VOICE_ALPHA = 0.65 # project gradio demo default
    DEFAULT_TEMPERATURE = 0.8 # project api default
    DEFAULT_USE_FP16 = False # project api default

    def generate(
            self,
            text: str,
            voice_path: str,
            temperature: float,
            emo_alpha: float,
            emo_voice_path: str,
            emo_vector: list[float]
    ) -> Sound | str:
        ...

class IndexTts2ModelProtocol(TtsModel, IndexTts2Protocol):
    ...

