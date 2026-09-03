import os
import random
from typing import Any

import numpy as np

from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types import DeviceType, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.util import *

import torch
from pocket_tts import TTSModel  # type: ignore
from pocket_tts.utils.config import CONFIGS_DIR  # type: ignore
# Internal library helpers, used to mirror Pocket's own resolution of voice
# references (predefined names / URLs) so the shared cache key can point at
# the on-disk file the library ends up using.
from pocket_tts.utils.utils import (  # type: ignore
    _ORIGINS_OF_PREDEFINED_VOICES,
    download_if_necessary,
    get_predefined_voice,
)
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


class PocketModel(PocketBaseModel):

    # The voice state is large and lives on the model's device, so only the
    # most recently prepared voice is retained (the base cache evicts the
    # previous one when a different voice is selected).
    RETAINS_MULTIPLE_VOICE_CLONES = False

    def __init__(self, device: DeviceType, language: str = ""):

        # Rem, "language" dictates model
        self.model: TTSModel | None = TTSModel.load_model(language=language or None)
        assert self.model

        self._device_type = device
        self.model.to(device.value)

    def kill(self) -> None:
        self.clear_voice_clone_cache()
        self.model = None

    def _resolve_voice_source_path(self, voice_path: str) -> str:
        """
        Pocket voice references can be local files, but also bare
        predefined-voice names, hf:// references or URLs, which the library
        resolves and downloads internally. The shared cache key requires an
        on-disk file, so resolve to the local file the library ends up using
        (its mtime/size then track the actual cached content).
        """
        if os.path.isfile(voice_path):
            return voice_path

        assert self.model

        if voice_path in _ORIGINS_OF_PREDEFINED_VOICES:
            # Same guard as the library: predefined voices need a language
            # config to resolve their per-language embedding file.
            origin = self.model.origin
            if origin is None or not origin.is_relative_to(CONFIGS_DIR):
                raise ValueError(
                    f"Cannot use predefined voices when the model "
                    f"is not loaded from a config associated with a language. "
                    f"Here the origin is {origin}"
                )
            ref = get_predefined_voice(language=origin.stem, name=voice_path)
        else:
            ref = voice_path

        return str(download_if_necessary(ref))

    def _create_voice_clone(self, voice_path: str) -> Any:
        """
        Pocket voice-state construction is expensive because it encodes and
        prompts the reference audio before text generation can begin. The
        base cache reuses the computed state while the source file (and its
        mtime/size) is unchanged.

        ``voice_path`` is the original reference as given by the caller (the
        library resolves predefined names / URLs itself); the cache key uses
        the resolved on-disk file instead (see _resolve_voice_source_path).

        The state is kept on the model's device on purpose: Pocket retains a
        single voice at a time, so this matches the previous memory profile.

        Reminder: this relies on Pocket's generate_audio_stream() default
        copy_state=True behavior. If a future change uses copy_state=False,
        generation may mutate the passed state and this cache would need to
        be revisited.

        Raising here aborts the generation with an error string (handled by
        the caller) and nothing gets cached.
        """
        assert self.model
        voice_state = self.model.get_state_for_audio_prompt(voice_path)
        device = self.model.device
        for module_state in voice_state.values():
            for k, v in module_state.items():
                module_state[k] = v.to(device)
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
            voice_selection_index: int = 0,
    ) -> list[Sound] | str:
        voice_file_name = ProjectVoiceUtil.current_voice_value(project, TtsModelType.POCKET, voice_selection_index)
        if voice_file_name:
            voice_path = ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name)
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

            if voice_path:
                try:
                    voice_state = self._get_or_create_voice_clone(
                        source_path=self._resolve_voice_source_path(voice_path),
                        transcript="",
                        factory=lambda: self._create_voice_clone(voice_path),
                    )
                except Exception as e:
                    return f"Couldn't create voice clone for {voice_path} - {make_error_string(e)}"
            else:
                # Keep the original (uncached) behavior for a missing voice path
                voice_state = self.model.get_state_for_audio_prompt(voice_path)

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
                sounds.append(Sound(audio_np, self.INFO.default_output_sample_rate))

            if on_stream_end is not None:
                on_stream_end()

            return sounds
        except Exception as e:
            return make_error_string(e)
