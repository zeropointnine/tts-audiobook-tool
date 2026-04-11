import os
import random
import traceback
import numpy as np
import torch
import chatterbox.mtl_tts # type: ignore
from chatterbox.mtl_tts import ChatterboxMultilingualTTS # type: ignore
from chatterbox.tts_turbo import ChatterboxTurboTTS # type: ignore

import logging

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
logging.getLogger("transformers").setLevel(logging.ERROR)

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import make_error_string


class ChatterboxModel(ChatterboxBaseModel):
    """
    Chatterbox inference logic
    """

    def __init__(self, model_type: ChatterboxType, device: str):
        self._device = device
        device_obj = torch.device(self._device)
        self._model_type = model_type

        match self._model_type:
            case ChatterboxType.MULTILINGUAL:
                self._chatterbox = ChatterboxMultilingualTTS.from_pretrained(device=device_obj)
            case ChatterboxType.TURBO:
                self._chatterbox = ChatterboxTurboTTS.from_pretrained(device=device_obj)

    def supported_languages_multi(self) -> list[str]:
        return list(chatterbox.mtl_tts.SUPPORTED_LANGUAGES)

    def kill(self) -> None:
        self._chatterbox = None # type: ignore

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False
        ) -> list[Sound] | str:
        
        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")
        prompt = prompts[0]

        if project.chatterbox_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
        else:
            voice_path = ""

        if project.chatterbox_type == ChatterboxType.MULTILINGUAL:
            language_id = project.language_code 
        else:
            language_id = ""

        exaggeration = project.chatterbox_exaggeration
        if exaggeration == -1:
            exaggeration = ChatterboxBaseModel.DEFAULT_EXAGGERATION

        cfg = project.chatterbox_cfg
        if cfg == -1:
            cfg = ChatterboxBaseModel.DEFAULT_CFG

        temperature = project.chatterbox_temperature
        if temperature == -1:
            temperature = ChatterboxBaseModel.DEFAULT_TEMPERATURE

        top_p = project.chatterbox_top_p
        if top_p == -1:
            top_p = ChatterboxBaseModel.DEFAULT_TOP_P

        if project.chatterbox_type == ChatterboxType.TURBO:
            turbo_top_k = project.chatterbox_turbo_top_k
        else:
            turbo_top_k = -1

        if project.chatterbox_type == ChatterboxType.MULTILINGUAL:
            repetition_penalty = project.chatterbox_ml_repetition_penalty
            if repetition_penalty == -1:
                repetition_penalty = ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_ML
        else:
            repetition_penalty = project.chatterbox_turbo_repetition_penalty
            if repetition_penalty == -1:
                repetition_penalty = ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_TURBO

        seed = -1 if force_random_seed else project.chatterbox_seed

        result = self.generate(
            text=prompt,
            voice_path=voice_path,
            exaggeration=exaggeration,
            cfg=cfg,
            temperature=temperature,
            top_p=top_p,
            turbo_top_k=turbo_top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            language_id=language_id
        )

        if isinstance(result, Sound):
            return [result]
        else:
            return result

    def generate(
        self,
        text: str,
        voice_path: str,
        exaggeration: float,
        cfg: float,
        temperature: float,
        top_p: float,
        turbo_top_k: int,
        repetition_penalty: float,
        seed: int,
        language_id: str
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
            seed = random.randrange(0, 2**32 - 1)
        AppUtil.set_seed(seed)

        dic = {}
        if language_id:
            dic["language_id"] = language_id
        if voice_path:
            dic["audio_prompt_path"] = voice_path        
        dic["exaggeration"] = exaggeration
        dic["cfg_weight"] = cfg
        dic["temperature"] = temperature
        dic["top_p"] = top_p
        if turbo_top_k != -1:
            dic["top_k"] = turbo_top_k # rem, multilingual does not support this param
        dic["repetition_penalty"] = repetition_penalty

        try:
            data = self._chatterbox.generate(text, **dic)
            data = data.cpu().numpy().squeeze()
            return Sound(data, TtsModelInfos.CHATTERBOX.value.sample_rate)
        except Exception as e:
            traceback.print_exc()
            return make_error_string(e)
        
