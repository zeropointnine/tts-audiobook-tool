import os
import random
import traceback
from typing import Any
import chatterbox.mtl_tts # type: ignore
from chatterbox.mtl_tts import ChatterboxMultilingualTTS # type: ignore
from chatterbox.tts_turbo import ChatterboxTurboTTS # type: ignore

import logging

from tts_audiobook_tool import app_support
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
logging.getLogger("transformers").setLevel(logging.ERROR)

from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_error_string


class ChatterboxModel(ChatterboxBaseModel):
    """
    Chatterbox inference logic
    """

    def __init__(self, model_type: ChatterboxType, device: str):
        
        self._device = device
        self._model_type = model_type
        
        multilingual_loader: Any = ChatterboxMultilingualTTS
        turbo_loader: Any = ChatterboxTurboTTS

        match self._model_type:
            case ChatterboxType.MULTILINGUAL:
                # Pass the normalized device string instead of torch.device(...).
                # Upstream Chatterbox checks for values like "cpu" and "mps"
                # before deciding whether to remap CUDA-saved checkpoints to CPU.
                self._chatterbox = multilingual_loader.from_pretrained(device=self._device)
            case ChatterboxType.TURBO:
                self._chatterbox = turbo_loader.from_pretrained(device=self._device)

    def supported_languages_multi(self) -> list[str]:
        return list(chatterbox.mtl_tts.SUPPORTED_LANGUAGES)

    def kill(self) -> None:
        self._chatterbox = None # type: ignore

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
        ) -> list[Sound] | str:
        
        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")

        # Parameters common to both model types
        dic = {
            "text": prompts[0],
            "voice_path": os.path.join(project.dir_path, project.chatterbox_voice_file_name)
                if project.chatterbox_voice_file_name else "",
            "temperature": project.chatterbox_temperature
                if project.chatterbox_temperature != -1 else ChatterboxBaseModel.DEFAULT_TEMPERATURE,
            "top_p": project.chatterbox_top_p
                if project.chatterbox_top_p != -1 else ChatterboxBaseModel.DEFAULT_TOP_P,
            "seed": -1 if force_random_seed else project.chatterbox_seed,
        }
        match self._model_type:
            case ChatterboxType.MULTILINGUAL:
                dic.update({
                    "language_id": project.language_code,
                    "exaggeration": project.chatterbox_exaggeration
                        if project.chatterbox_exaggeration != -1 else ChatterboxBaseModel.DEFAULT_EXAGGERATION,
                    "cfg": project.chatterbox_cfg
                        if project.chatterbox_cfg != -1 else ChatterboxBaseModel.DEFAULT_CFG,
                    "repetition_penalty": project.chatterbox_ml_repetition_penalty
                        if project.chatterbox_ml_repetition_penalty != -1 else ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_ML,
                })
            case ChatterboxType.TURBO:
                dic.update({
                    "turbo_top_k": project.chatterbox_turbo_top_k,
                    "repetition_penalty": project.chatterbox_turbo_repetition_penalty
                        if project.chatterbox_turbo_repetition_penalty != -1 else ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_TURBO,
                })
            # Note how each model has an independent repetition penalty value b/c the values behave differently on each

        result = self.generate(**dic)

        if isinstance(result, Sound):
            return [result]
        else:
            return result

    def generate(
        self,
        text: str,
        voice_path: str,
        exaggeration: float = -1,
        cfg: float = -1,
        temperature: float = ChatterboxBaseModel.DEFAULT_TEMPERATURE,
        top_p: float = ChatterboxBaseModel.DEFAULT_TOP_P,
        turbo_top_k: int = -1,
        repetition_penalty: float = -1,
        seed: int = -1,
        language_id: str = ""
    ) -> Sound | str:
        """
        :param seed: If -1, is set to random int
        :param turbo_top_k: If -1, is not passed to model
        """

        if self._chatterbox is None:
            return "Logic error: Model is not initialized"
        if language_id and self._model_type == ChatterboxType.TURBO:
            return "Logic error: language_id is not supported for Chatterbox Turbo"
        
        if seed <= -1:
            seed = random.randrange(0, SEED_MAX)
        app_support.set_seed(seed)

        if repetition_penalty == -1:
            match self._model_type:
                case ChatterboxType.MULTILINGUAL:
                    repetition_penalty = ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_ML
                case ChatterboxType.TURBO:
                    repetition_penalty = ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_TURBO

        dic = {}
        if voice_path:
            dic["audio_prompt_path"] = voice_path        
        dic["temperature"] = temperature
        dic["top_p"] = top_p
        dic["repetition_penalty"] = repetition_penalty

        match self._model_type:
            case ChatterboxType.MULTILINGUAL:
                if language_id:
                    dic["language_id"] = language_id
                if exaggeration == -1:
                    exaggeration = ChatterboxBaseModel.DEFAULT_EXAGGERATION
                if cfg == -1:
                    cfg = ChatterboxBaseModel.DEFAULT_CFG
                dic["exaggeration"] = exaggeration
                dic["cfg_weight"] = cfg
            case ChatterboxType.TURBO:
                if turbo_top_k != -1:
                    dic["top_k"] = turbo_top_k # rem, multilingual does not support this param

        try:
            data = self._chatterbox.generate(text, **dic)
            data = data.cpu().numpy().squeeze()
            return Sound(data, TtsModelType.CHATTERBOX.value.sample_rate)
        except Exception as e:
            traceback.print_exc()
            return make_error_string(e)
        
