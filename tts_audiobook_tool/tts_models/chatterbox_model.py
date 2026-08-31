import os
import random
import traceback
from typing import Any
import torch
import chatterbox.mtl_tts # type: ignore
from chatterbox.mtl_tts import ChatterboxMultilingualTTS # type: ignore
from chatterbox.models.t3.modules.cond_enc import T3Cond # type: ignore
from chatterbox.tts_turbo import ChatterboxTurboTTS # type: ignore

import logging

from tts_audiobook_tool import app_support
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
logging.getLogger("transformers").setLevel(logging.ERROR)

from tts_audiobook_tool.app_types import DeviceType, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_error_string
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


class ChatterboxModel(ChatterboxBaseModel):
    """
    Chatterbox inference logic
    """

    # Prepared conditionals are CPU-cloned (see `_create_voice_clone`), so
    # retaining several voices at once costs only a few MB of RAM per voice.
    RETAINS_MULTIPLE_VOICE_CLONES = True

    def __init__(self, model_type: ChatterboxType, device: DeviceType):

        self._device_type = device
        device_value = device.value
        self._model_type = model_type

        multilingual_loader: Any = ChatterboxMultilingualTTS
        turbo_loader: Any = ChatterboxTurboTTS

        match self._model_type:
            case ChatterboxType.MULTILINGUAL:
                # Pass the normalized device string instead of torch.device(...).
                # Upstream Chatterbox checks for values like "cpu" and "mps"
                # before deciding whether to remap CUDA-saved checkpoints to CPU.
                self._chatterbox = multilingual_loader.from_pretrained(device=device_value)
            case ChatterboxType.TURBO:
                self._chatterbox = turbo_loader.from_pretrained(device=device_value)

    def supported_languages_multi(self) -> list[str]:
        return list(chatterbox.mtl_tts.SUPPORTED_LANGUAGES)

    def kill(self) -> None:
        self.clear_voice_clone_cache()
        self._chatterbox = None # type: ignore

    def _create_voice_clone(self, source_path: str) -> Any:
        """
        Prepares the conditionals for the given voice file and returns a
        CPU clone of them.

        The expensive work (reference feature embedding, speech prompt
        tokens, voice-encoder speaker embedding) happens here, once per
        file state. The library's own `self.conds` is left pointing at the
        on-device objects it created; the cache keeps CPU copies only.
        """
        self._chatterbox.prepare_conditionals(source_path) # type: ignore
        conds = self._chatterbox.conds # type: ignore
        if conds is None:
            raise RuntimeError("Chatterbox prepare_conditionals() did not produce conditionals")
        return self._clone_conditionals(conds, device="cpu")

    @staticmethod
    def _clone_conditionals(conds: Any, device: str) -> Any:
        """
        Builds a fresh Conditionals-like object whose tensors live on
        `device`, without mutating the input.

        The library's Conditionals.to() mutates in place, which would move
        the cached value off CPU, so a new object is built instead. The
        multilingual and turbo modules each define their own Conditionals
        class, so the result uses the class of the input object.
        """
        t3 = conds.t3
        t3_copy = T3Cond(
            speaker_emb=ChatterboxModel._copy_tensor(t3.speaker_emb, device),
            clap_emb=ChatterboxModel._copy_tensor(t3.clap_emb, device),
            cond_prompt_speech_tokens=ChatterboxModel._copy_tensor(t3.cond_prompt_speech_tokens, device),
            cond_prompt_speech_emb=ChatterboxModel._copy_tensor(t3.cond_prompt_speech_emb, device),
            emotion_adv=ChatterboxModel._copy_tensor(t3.emotion_adv, device),
        )
        gen_copy = {
            key: ChatterboxModel._copy_tensor(value, device) if torch.is_tensor(value) else value
            for key, value in conds.gen.items()
        }
        return type(conds)(t3_copy, gen_copy)

    @staticmethod
    def _copy_tensor(value: Any, device: str) -> Any:
        if value is None or not torch.is_tensor(value):
            return value
        return value.detach().to(device).clone()

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_selection_index: int = 0,
        ) -> list[Sound] | str:

        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")

        # Parameters common to both model types
        voice_file_name = ProjectVoiceUtil.current_voice_value(project, TtsModelType.CHATTERBOX, voice_selection_index)

        dic = {
            "text": prompts[0],
            "voice_path": os.path.join(project.dir_path, voice_file_name) if voice_file_name else "",
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
            # Get-or-create the prepared conditionals, then hand the library
            # a fresh on-device copy (the library may mutate `self.conds`,
            # so the cached CPU value must not be shared with it).
            try:
                conds = self._get_or_create_voice_clone(
                    source_path=voice_path,
                    transcript="",
                    factory=lambda: self._create_voice_clone(voice_path),
                )
            except Exception as e:
                return f"Couldn't create voice clone for {voice_path} - {make_error_string(e)}"
            self._chatterbox.conds = self._clone_conditionals(conds, self._chatterbox.device)
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

