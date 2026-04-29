# tts_audiobook_tool/tts_model/omnivoice_base_model.py
from __future__ import annotations

from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class OmniVoiceBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.OMNIVOICE.value

    DEFAULT_REPO_ID = "k2-fsa/OmniVoice"
    SAMPLE_RATE     = 24_000   # OmniVoice's default generation frequency
    DEFAULT_SPEED   = 1.0

    def __init__(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        text: str,
        ref_audio_path: str = "",
        ref_text: str = "",
        speed: float = DEFAULT_SPEED,
        instruct: str = "",
        language_id: str = "",
        duration: float | None = None,
        **kwargs,
    ) -> tuple[list, int]:
        """
        Returns ([audio_array_float32_mono], sample_rate).
        """
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError
