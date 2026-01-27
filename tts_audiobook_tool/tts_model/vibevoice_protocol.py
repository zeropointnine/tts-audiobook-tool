from typing import Protocol
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class VibeVoiceProtocol(Protocol):

    DEFAULT_REPO_ID = "microsoft/VibeVoice-1.5b"

    # nb, their gradio demo default is 1.3, which is IMO much too low
    CFG_DEFAULT = 3.0
    CFG_MIN = 1.0
    CFG_MAX = 7.0

    DEFAULT_NUM_STEPS = 10 # from vibevoice library code

    # Must accommodate worst-case prompt size (app limit 80 words)
    MAX_TOKENS = 250

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            cfg_scale: float = 3.0,
            num_steps: int = 10,
            seed: int = -1
    ) -> list[Sound] | str:
        ...

    @property
    def has_lora(self) -> bool:
        ...

class VibeVoiceModelProtocol(TtsModel, VibeVoiceProtocol):
    ...

