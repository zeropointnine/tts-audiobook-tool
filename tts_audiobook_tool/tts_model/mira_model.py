import random
from itertools import cycle
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.mira_base_model import MiraBaseModel
from tts_audiobook_tool.util import *

import torch

from mira.model import MiraTTS # type: ignore


class MiraModel(MiraBaseModel):

    def __init__(self): 
        self.mira_tts = MiraTTS('YatharthS/MiraTTS')
        self.context_tokens = None
        self.last_voice_path: str = ""

    def set_voice_clone(self, path: str) -> None:
        assert(self.mira_tts is not None)
        if self.context_tokens and path == self.last_voice_path:
            return
        if not path:
            self.context_tokens = None
        else:
            self.context_tokens = self.mira_tts.encode_audio(path)
        self.last_voice_path = path

    def clear_voice_clone(self) -> None: 
        """ Important to call this manually due to 'last_voice_path' """
        self.context_tokens = None
        self.last_voice_path = ""

    def kill(self) -> None:
        self.mira_tts = None
        self.context_tokens = None

    def set_params(self, temperature: float, max_new_tokens: int, top_k: int=-1, top_p: float=-1, repetition_penalty: float=-1) -> None:
        assert(self.mira_tts is not None)
        self.mira_tts.set_params(temperature=temperature, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False
        ) -> list[Sound] | str:

        voice_path = os.path.join(project.dir_path, project.mira_voice_file_name)
        self.set_voice_clone(voice_path)

        if project.mira_temperature == -1:
            temperature = MiraBaseModel.TEMPERATURE_DEFAULT
        elif project.mira_temperature < MiraBaseModel.TEMPERATURE_MIN or project.mira_temperature > MiraBaseModel.TEMPERATURE_MAX:
            temperature = MiraBaseModel.TEMPERATURE_DEFAULT
        else:
            temperature = project.mira_temperature

        if project.mira_top_p == -1:
            top_p = MiraBaseModel.TOP_P_DEFAULT
        else:
            top_p = project.mira_top_p

        if project.mira_top_k == -1:
            top_k = MiraBaseModel.TOP_K_DEFAULT
        else:
            top_k = project.mira_top_k

        if project.mira_repetition_penalty == -1:
            repetition_penalty = MiraBaseModel.REPETITION_PENALTY_DEFAULT
        else:
            repetition_penalty = project.mira_repetition_penalty

        self.set_params(
            temperature=temperature, max_new_tokens=MiraBaseModel.MAX_NEW_TOKENS,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty
        )

        assert(self.mira_tts is not None)
        seed = -1 if force_random_seed else project.mira_seed
        if seed == -1:
            seed = random.randrange(0, 2**32 - 1)
        AppUtil.set_seed(seed)
        self.mira_tts.gen_config.random_seed = seed

        if len(prompts) == 1:
            result = self.generate_single(prompts[0])
            if isinstance(result, Sound):
                return [result]
            else:
                return result
        else:
            result = self.generate_batch(prompts)
            return result

    def generate_single(self, prompt: str) -> Sound | str:
        
        if not self.mira_tts:
            return "Logic error - model not initialized"
        if not self.context_tokens:
            return "Logic error - voice clone not set"
        
        try:
            # Model outputs float16 
            audio_tensor = self.mira_tts.generate(prompt, self.context_tokens) 
            # Convert to float32, put on cpu, numpy format
            audio_np = audio_tensor.to(torch.float32).cpu().numpy()
        except Exception as e:
            return make_error_string(e)

        return Sound(audio_np, MiraModel.INFO.sample_rate)


    def generate_batch(self, prompts: list[str]) -> list[Sound] | str:

        assert(self.mira_tts is not None)
        if not self.context_tokens:
            return "Logic error - voice clone not set"
        
        context_tokens = [self.context_tokens]

        formatted_prompts = []
        for prompt, context_token in zip(prompts, cycle(context_tokens)):
            formatted_prompt = self.mira_tts.codec.format_prompt(prompt, context_token, None)
            formatted_prompts.append(formatted_prompt)
        
        responses = self.mira_tts.pipe(formatted_prompts, gen_config=self.mira_tts.gen_config, do_preprocess=False)
        generated_tokens = [response.text for response in responses]
        
        sounds: list[Sound] = []
        
        for generated_token, context_token in zip(generated_tokens, cycle(context_tokens)):
            print(f"< {len(sounds) + 1}/{len(prompts)} >", end=Ansi.ERASE_REST_OF_LINE + Ansi.LINE_HOME, flush=True)
            audio = self.mira_tts.codec.decode(generated_token, context_token)
            audio_np = audio.to(torch.float32).cpu().numpy()
            sound = Sound(audio_np, MiraModel.INFO.sample_rate)
            sounds.append(sound)
        print()
                    
        return sounds
