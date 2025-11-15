from chatterbox.tts import ChatterboxTTS # type: ignore
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model import ChatterboxModelProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import make_error_string


class ChatterboxModel(ChatterboxModelProtocol):
    """
    Chatterbox inference logic
    """

    def __init__(self, device: str):
        super().__init__(info=TtsModelInfos.CHATTERBOX.value)
        self._chatterbox: ChatterboxTTS = ChatterboxTTS.from_pretrained(device=device)

    def kill(self) -> None:
        self._chatterbox = None # type: ignore

    def generate(
        self,
        text: str,
        voice_path: str = "",
        exaggeration: float = -1,
        cfg: float = -1,
        temperature: float = -1
    ) -> Sound | str:

        dic = {}
        if voice_path:
            dic["audio_prompt_path"] = voice_path
        if exaggeration != -1:
            dic["exaggeration"] = exaggeration
        if cfg != -1:
            dic["cfg_weight"] = cfg
        if temperature != -1:
            dic["temperature"] = temperature

        try:
            data = self._chatterbox.generate(text, **dic)
            data = data.numpy().squeeze()
            return Sound(data, self.info.sample_rate)
        except Exception as e:
            return make_error_string(e)
