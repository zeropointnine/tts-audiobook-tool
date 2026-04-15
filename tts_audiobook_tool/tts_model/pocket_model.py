import os
import random
import traceback

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.util import *

import torch
from pocket_tts import TTSModel  # type: ignore


class PocketModel(PocketBaseModel):

    def __init__(self, device: str, language: str = ""):

        self.model: TTSModel | None = TTSModel.load_model(language=language or None)
        self.model.to(device)

    def kill(self) -> None:
        self.model = None

    def generate_using_project(
            self, project: Project, prompts: list[str], force_random_seed: bool = False
    ) -> list[Sound] | str:
        if project.pocket_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.pocket_voice_file_name)
        else:
            voice_path = project.pocket_predefined_voice

        temperature = project.pocket_temperature
        if temperature == -1:
            temperature = PocketModel.DEFAULT_TEMPERATURE

        seed = -1 if force_random_seed else project.pocket_seed

        return self.generate(prompts, voice_path, temperature, seed)

    def generate(
            self, texts: list[str], voice_path: str, temperature: float, seed: int
    ) -> list[Sound] | str:
        
        # Print something to stay consistent w/ other model behavior b/c model lib does not 
        printt(f"{COL_DIM}{Ansi.ITALICS}Generating...")

        try:
            assert self.model
            self.model.temp = temperature

            self.model.lsd_decode_steps = PocketBaseModel.LSD

            voice_state = self.model.get_state_for_audio_prompt(voice_path)
            device = self.model.device
            for module_state in voice_state.values():
                for k, v in module_state.items():
                    module_state[k] = v.to(device)
            if seed <= -1:
                seed = random.randrange(0, SEED_MAX)
            AppUtil.set_seed(seed)
            sounds = []
            for text in texts:
                audio = self.model.generate_audio(voice_state, text, max_tokens=PocketBaseModel.MAX_TOKENS)
                audio_np = audio.to(torch.float32).cpu().numpy()
                sounds.append(Sound(audio_np, PocketModel.INFO.sample_rate))
            return sounds
        except Exception as e:
            return make_error_string(e)
