from itertools import cycle
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model import MiraModelProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *

import torch

from mira.model import MiraTTS # type: ignore


class MiraModel(MiraModelProtocol):

    def __init__(self): 
        super().__init__(info=TtsModelInfos.MIRA.value)

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

    def set_params(self, temperature: float, max_new_tokens: int) -> None:
        assert(self.mira_tts is not None)
        self.mira_tts.set_params(temperature=temperature, max_new_tokens=max_new_tokens)

    def generate(self, prompt: str) -> Sound | str:
        
        if not self.mira_tts:
            return "Logic error - model not initialized"
        if not self.context_tokens:
            return "Logic error - voice clone not set"
        
        # Defaults are:
        # self.mira_tts.set_params(
        #   top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024, repetition_penalty=1.2, min_p=0.05
        # )

        try:
            # Model outputs float16 
            audio_tensor = self.mira_tts.generate(prompt, self.context_tokens) 
            # Convert to float32, put on cpu, numpy format
            audio_np = audio_tensor.to(torch.float32).cpu().numpy()
        except Exception as e:
            return make_error_string(e)

        return Sound(audio_np, self.info.sample_rate)


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
            sound = Sound(audio_np, self.info.sample_rate)
            sounds.append(sound)
        print()
                    
        return sounds
