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

        if self._model_type.is_multilingual:
            # Pass the normalized device string instead of torch.device(...).
            # Upstream Chatterbox checks for values like "cpu" and "mps"
            # before deciding whether to remap CUDA-saved checkpoints to CPU.
            self._chatterbox = multilingual_loader.from_pretrained(
                device=device_value,
                t3_model=self._model_type.multilingual_t3_model,
            )
        else:
            self._chatterbox = turbo_loader.from_pretrained(device=device_value)

        if device == DeviceType.CUDA:
            ChatterboxModel._use_gpu_watermarker(self._chatterbox, device_value)

    @staticmethod
    def _use_gpu_watermarker(chatterbox: Any, device_value: str) -> None:
        """
        Replaces the library's CPU Perth watermarker with a GPU instance.

        Chatterbox watermarks every generated segment via
        `perth.PerthImplicitWatermarker().apply_watermark()`. Perth's CPU
        compute path leaks native (non-Python) memory on every call —
        measured at roughly 5-20 MB per call depending on audio length,
        linearly, and never released: Python GC, torch empty_cache, thread
        count, and watermarker-instance recycling all make no difference;
        only process exit frees it (which is why `Options > Unload models`
        appears to reclaim it — it terminates the worker). The GPU path does
        not leak, and the PerthNet model is tiny, so the extra VRAM cost is
        negligible. CPU-watermarked and GPU-watermarked audio decode to the
        same watermark bits. (Upstream perth bug; revisit if fixed.)
        """
        try:
            import perth # type: ignore
            chatterbox.watermarker = perth.PerthImplicitWatermarker(
                device=device_value
            )
        except Exception:
            traceback.print_exc()

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

    @staticmethod
    def _resolve_setting(value: float, default: float) -> float:
        """
        Maps the project-level "unset" sentinel (-1) to the given default.
        """
        if value == -1:
            return default
        return value

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_selection_index: int = 0,
            print_params: bool = False,
        ) -> list[Sound] | str:

        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")

        # Parameters common to both model types
        voice_file_name = ProjectVoiceUtil.current_voice_value(project, TtsModelType.CHATTERBOX, voice_selection_index)

        temperature = ChatterboxModel._resolve_setting(project.chatterbox_temperature, ChatterboxBaseModel.DEFAULT_TEMPERATURE)
        top_p = ChatterboxModel._resolve_setting(project.chatterbox_top_p, ChatterboxBaseModel.DEFAULT_TOP_P)
        # Only consumed by the multilingual variant, but always passed
        exaggeration = ChatterboxModel._resolve_setting(project.chatterbox_exaggeration, ChatterboxBaseModel.DEFAULT_EXAGGERATION)
        cfg = ChatterboxModel._resolve_setting(project.chatterbox_cfg, ChatterboxBaseModel.DEFAULT_CFG)

        # Note how each model has an independent repetition penalty value b/c the values behave differently on each
        language_id = ""
        repetition_penalty: float
        turbo_top_k: int | None = None
        if self._model_type.is_multilingual:
            language_id = project.language_code
            ml_repetition_penalty = (
                project.chatterbox_ml_v2_repetition_penalty
                if self._model_type == ChatterboxType.MULTILINGUAL_V2
                else project.chatterbox_ml_v3_repetition_penalty
            )
            repetition_penalty = ChatterboxModel._resolve_setting(
                ml_repetition_penalty,
                ChatterboxBaseModel.default_repetition_penalty(self._model_type),
            )
        else:
            turbo_top_k = None if project.chatterbox_turbo_top_k == -1 else project.chatterbox_turbo_top_k
            repetition_penalty = ChatterboxModel._resolve_setting(
                project.chatterbox_turbo_repetition_penalty, ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_TURBO
            )

        # Randomize seed here so that generate() receives a concrete value
        seed = -1 if force_random_seed else project.chatterbox_seed
        if seed <= -1:
            seed = random.randrange(0, SEED_MAX)

        result = self.generate(
            text=prompts[0],
            voice_path=ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name) if voice_file_name else "",
            temperature=temperature,
            top_p=top_p,
            exaggeration=exaggeration,
            cfg=cfg,
            seed=seed,
            language_id=language_id,
            repetition_penalty=repetition_penalty,
            turbo_top_k=turbo_top_k,
            print_params=print_params
        )

        if isinstance(result, Sound):
            return [result]
        else:
            return result

    def generate(
        self,
        text: str,
        voice_path: str,
        repetition_penalty: float,
        seed: int,
        exaggeration: float,
        cfg: float,
        temperature: float,
        top_p: float,
        turbo_top_k: int | None = None,
        language_id: str = "",
        print_params: bool=False
    ) -> Sound | str:
        """
        All values must be concrete (ie, resolved defaults, randomized seed).
        `exaggeration`/`cfg` are only consumed by the multilingual variant.
        :param turbo_top_k: If None, is not passed to the model
        """

        if self._chatterbox is None:
            return "Logic error: Model is not initialized"
        if language_id and self._model_type == ChatterboxType.TURBO:
            return "Logic error: language_id is not supported for Chatterbox Turbo"

        if print_params:
            from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
            TtsBaseModel.print_params(locals())

        app_support.set_seed(seed)

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

        if self._model_type.is_multilingual:
            if language_id:
                dic["language_id"] = language_id
            dic["exaggeration"] = exaggeration
            dic["cfg_weight"] = cfg
        elif turbo_top_k is not None:
            dic["top_k"] = turbo_top_k # rem, multilingual does not support this param

        try:
            data = self._chatterbox.generate(text, **dic)
            data = data.cpu().numpy().squeeze()
            return Sound(data, self.INFO.default_output_sample_rate)
        except Exception as e:
            traceback.print_exc()
            return make_error_string(e)

