import os
import random

import numpy as np

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.util import *

import torch
from pocket_tts import TTSModel  # type: ignore


class PocketModel(PocketBaseModel):

    def __init__(self, device: str, language: str = ""):

        # Rem, "language" dictates model
        self.model: TTSModel | None = TTSModel.load_model(language=language or None)
        assert self.model
        self.model.to(device)
        self.last_voice_path = ""
        self.cached_voice_state = None

    def kill(self) -> None:
        self.model = None

    def get_voice_state(self, voice_path: str):
        assert self.model

        # Pocket voice-state construction is expensive because it encodes and
        # prompts the reference audio before text generation can begin. Reuse
        # the computed state when the same voice path is used again.
        #
        # Reminder: this relies on Pocket's generate_audio_stream() default
        # copy_state=True behavior. If a future change uses copy_state=False,
        # generation may mutate the passed state and this cache would need to
        # be revisited.
        if voice_path == self.last_voice_path and self.cached_voice_state is not None:
            return self.cached_voice_state

        voice_state = self.model.get_state_for_audio_prompt(voice_path)
        device = self.model.device
        for module_state in voice_state.values():
            for k, v in module_state.items():
                module_state[k] = v.to(device)

        self.last_voice_path = voice_path
        self.cached_voice_state = voice_state
        return voice_state

    def get_voice_clone_access_error_for_path(self, voice_path: str) -> str:
        try:
            assert self.model
            _ = self.model.get_state_for_audio_prompt(voice_path)
            return ""
        except Exception as e:
            return make_error_string(e)

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:
        if project.pocket_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.pocket_voice_file_name)
        else:
            voice_path = project.pocket_predefined_voice

        temperature = project.pocket_temperature
        if temperature == -1:
            temperature = PocketModel.DEFAULT_TEMPERATURE

        seed = -1 if force_random_seed else project.pocket_seed

        return self.generate(
            prompts,
            voice_path,
            temperature,
            seed,
            on_stream_chunk=on_stream_chunk,
            on_stream_end=on_stream_end,
        )

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            temperature: float,
            seed: int,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:
        
        # Print something to stay consistent w/ other model behavior b/c model lib does not 
        printt(f"{COL_DIM_ITALICS}Generating...")

        try:
            assert self.model
            self.model.temp = temperature

            self.model.lsd_decode_steps = PocketBaseModel.LSD

            voice_state = self.get_voice_state(voice_path)
            if seed <= -1:
                seed = random.randrange(0, SEED_MAX)
            app_support.set_seed(seed)
            sounds = []
            for text in texts:
                audio_chunks = []

                # Pocket can internally split long text into multiple
                # chunks/sub-generations when max_tokens is exceeded. We
                # intentionally ignore that distinction here and treat the
                # outer generate_audio_stream() iterator as one prompt stream,
                # because this app sets PocketBaseModel.MAX_TOKENS high enough
                # that Pocket's internal text chunking is not expected for our
                # normal prompt sizes.
                for chunk in self.model.generate_audio_stream(
                    voice_state,
                    text,
                    max_tokens=PocketBaseModel.MAX_TOKENS,
                ):
                    chunk_np = chunk.to(torch.float32).cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                    audio_chunks.append(chunk_np)

                    if on_stream_chunk is not None:
                        on_stream_chunk(chunk_np)

                if not audio_chunks:
                    return "No audio output"

                audio_np = np.concatenate(audio_chunks)
                sounds.append(Sound(audio_np, PocketModel.INFO.sample_rate))

            if on_stream_end is not None:
                on_stream_end()

            return sounds
        except Exception as e:
            return make_error_string(e)
