import copy
import outetts # type: ignore

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model import OuteModelProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import make_error_string


class OuteModel(OuteModelProtocol):
    """
    Oute inference logic
    """

    def __init__(self):
        super().__init__(info=TtsModelInfos.OUTE.value)

        from tts_audiobook_tool.config_oute import MODEL_CONFIG # let app crash on error
        try:
            # Overwrite with dev version if exists
            from ..config_oute_dev import MODEL_CONFIG # type: ignore
        except ImportError:
            pass
        self._interface = outetts.Interface(config=MODEL_CONFIG)

        from loguru import logger
        logger.remove()

    def create_speaker(self, path: str) -> dict | str:
        """
        Returns voice dict or error string
        """
        try:
            assert(self._interface is not None)
            voice_dict = self._interface.create_speaker(path)
            return voice_dict
        except Exception as e:
            return make_error_string(e)

    def kill(self) -> None:
        self._interface = None # type: ignore

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False
        ) -> list[Sound] | str:
        
        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")
        prompt = prompts[0]

        result = self.generate(
            prompt,
            project.oute_voice_json,
            project.oute_temperature)

        if isinstance(result, Sound):
            return [result]
        else:
            return result
        
    def generate(
        self,
        prompt: str,
        voice: dict,
        temperature: float = -1
    ) -> Sound | str:
        """
        :param voice: Oute-specific voice clone data
        """

        # First, clone GENERATION_CONFIG from config file
        from outetts.models.config import SamplerConfig # type: ignore
        from tts_audiobook_tool.config_oute import GENERATION_CONFIG
        try:
            from ..config_oute_dev import GENERATION_CONFIG # type: ignore
        except ImportError:
            pass
        gen_config = copy.deepcopy(GENERATION_CONFIG)

        gen_config.text = prompt
        gen_config.speaker = voice
        if temperature != -1:
            gen_config.sampler_config = SamplerConfig(temperature)

        try:
            assert(self._interface is not None)
            output = self._interface.generate(config=gen_config)
            audio = output.audio.cpu().clone().squeeze().numpy()
            return Sound(audio, output.sr)

        except Exception as e:
            return make_error_string(e)
