import random

import numpy as np
import torch

from omnivoice import OmniVoice  # type: ignore
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig  # type: ignore
from omnivoice.models.omnivoice import VoiceClonePrompt  # type: ignore

from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types import DeviceType, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


class OmniVoiceModel(OmniVoiceBaseModel):
    """
    App wrapper for `omnivoice.OmniVoice`
    Supports Voice Cloning, Voice Design and Auto Voice modes.
    """

    # Effectively disable OmniVoice's own long-text chunker
    AUDIO_CHUNK_THRESHOLD_SECONDS = 999.0

    # Voice clone prompts are small (C x T int codes) and CPU-retained, so
    # several voices can be kept at once.
    RETAINS_MULTIPLE_VOICE_CLONES = True

    def __init__(self, model_target: str, device: DeviceType):

        self._model_target = model_target
        self._device_type = device

        if device == DeviceType.CUDA:
            device_map = "cuda:0"
        else:
            device_map = device.value  # "mps" or "cpu"

        if device == DeviceType.CUDA:
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._model: OmniVoice = OmniVoice.from_pretrained(
            model_target,
            device_map=device_map,
            dtype=dtype,
        )

    def kill(self) -> None:
        model = self._model

        try:
            if model is not None:
                asr_pipe = getattr(model, "_asr_pipe", None)
                if asr_pipe is not None:
                    asr_model = getattr(asr_pipe, "model", None)
                    if asr_model is not None and hasattr(asr_model, "cpu"):
                        asr_model.cpu()
                    model._asr_pipe = None

                audio_tokenizer = getattr(model, "audio_tokenizer", None)
                if audio_tokenizer is not None and hasattr(audio_tokenizer, "cpu"):
                    audio_tokenizer.cpu()

                if hasattr(model, "cpu"):
                    model.cpu()

                model.audio_tokenizer = None
                model.text_tokenizer = None
                model.feature_extractor = None # type: ignore
                model.duration_estimator = None
                model.sampling_rate = None
                model.llm = None
                model.audio_embeddings = None # type: ignore
                model.audio_heads = None # type: ignore
                model.codebook_layer_offsets = None # type: ignore
                model.normalized_audio_codebook_weights = None # type: ignore
        except Exception as e:
            L.e(f"{e}")

        self.clear_voice_clone_cache()
        self._model = None  # type: ignore

    # ── Main interface ────────────────────────────────────────────────

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_selection_index: int = 0,
            print_params: bool = False,
            max_random_seed: int = -1,
    ) -> list[Sound] | str:

        voice_file_name, ref_text = ProjectVoiceUtil.current_voice_reference_pair(
            project, TtsModelType.require_by_id("omnivoice_local"), voice_selection_index
        )
        voice_path = ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name) if voice_file_name else ""
        instruct = project.get_model_setting('omnivoice_local', 'instruct')
        cfg      = project.get_model_setting('omnivoice_local', 'cfg') if project.get_model_setting('omnivoice_local', 'cfg') != -1 else self.CFG_DEFAULT
        speed    = project.get_model_setting('omnivoice_local', 'speed') if project.get_model_setting('omnivoice_local', 'speed') != -1 else self.DEFAULT_SPEED
        steps    = project.get_model_setting('omnivoice_local', 'num_step') if project.get_model_setting('omnivoice_local', 'num_step') != -1 else self.DEFAULT_STEPS
        seed     = -1 if force_random_seed else project.get_model_setting('omnivoice_local', 'seed')

        has_voice    = bool(voice_path and os.path.isfile(voice_path))
        has_instruct = bool(instruct)

        # Common to all generation modes
        if seed == -1:
            seed = random.randrange(0, get_random_seed_max(SEED_MAX - 1, max_random_seed) + 1)
        generation_config = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=cfg,
            audio_chunk_threshold=self.AUDIO_CHUNK_THRESHOLD_SECONDS,
        )

        if has_voice:
            return self._generate_voice_clone(
                prompts=prompts,
                language=project.language_code,
                voice_path=voice_path,
                ref_text=ref_text,
                instruct=instruct,
                speed=speed,
                seed=seed,
                generation_config=generation_config,
                print_params=print_params,
            )
        elif has_instruct:
            return self._generate_voice_design(
                prompts=prompts,
                language=project.language_code,
                instruct=instruct,
                speed=speed,
                seed=seed,
                generation_config=generation_config,
                print_params=print_params,
            )
        else:
            return self._generate_auto_voice(
                prompts=prompts,
                language=project.language_code,
                speed=speed,
                seed=seed,
                generation_config=generation_config,
                print_params=print_params,
            )

    def _create_voice_clone(self, source_path: str, transcript: str) -> VoiceClonePrompt:
        """
        Creates the library's reusable prompt from the reference audio and
        returns a CPU clone of it.

        Raising here aborts the generation with an error string (handled by
        the caller) and nothing gets cached.
        """
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=source_path,
            ref_text=transcript or None,
        )
        return VoiceClonePrompt(
            ref_audio_tokens=prompt.ref_audio_tokens.detach().cpu().clone(),
            ref_text=prompt.ref_text,
            ref_rms=prompt.ref_rms,
        )

    # ── Generation modes ──────────────────────────────────────────────────

    def _generate_voice_clone(
            self,
            prompts: list[str],
            voice_path: str,
            ref_text: str,
            instruct: str,
            speed: float,
            seed: int,
            generation_config: OmniVoiceGenerationConfig,
            language: str,
            print_params: bool = False,
    ) -> list[Sound] | str:

        locals_snapshot = locals()

        try:
            # The library moves the (CPU) prompt tokens to its device at
            # generate time and never mutates them, so the cached prompt can
            # be used directly.
            voice_clone_prompt = self._get_or_create_voice_clone(
                source_path=voice_path,
                transcript=ref_text,
                factory=lambda: self._create_voice_clone(voice_path, ref_text),
            )
        except Exception as e:
            return f"Couldn't create voice clone for {voice_path} - {make_error_string(e)}"

        if print_params:
            self._print_params(locals_snapshot, generation_config)

        printt("Generating...", dont_reset=True)

        app_support.set_seed(seed)

        results = []
        for prompt in prompts:
            try:
                kw: dict = dict(
                    text=prompt,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    speed=speed,
                    generation_config=generation_config,
                )
                if instruct:
                    kw["instruct"] = instruct   # cloning + combined styles

                audio_arrays: list[np.ndarray] = self._model.generate(**kw)
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.INFO.default_output_sample_rate))

            except Exception as e:
                return make_error_string(e)

        return results

    def _generate_voice_design(
            self,
            prompts: list[str],
            instruct: str,
            speed: float,
            seed: int,
            generation_config: OmniVoiceGenerationConfig,
            language: str,
            print_params: bool = False,
    ) -> list[Sound] | str:

        if print_params:
            self._print_params(locals(), generation_config)

        printt("Generating...", dont_reset=True)

        app_support.set_seed(seed)

        results = []
        for prompt in prompts:
            try:
                audio_arrays: list[np.ndarray] = self._model.generate(
                    text=prompt,
                    language=language,
                    instruct=instruct,
                    speed=speed,
                    generation_config=generation_config,
                )
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.INFO.default_output_sample_rate))

            except Exception as e:
                return make_error_string(e)

        return results

    def _generate_auto_voice(
            self,
            prompts: list[str],
            speed: float,
            seed: int,
            generation_config: OmniVoiceGenerationConfig,
            language: str,
            print_params: bool = False,
    ) -> list[Sound] | str:

        if print_params:
            self._print_params(locals(), generation_config)

        printt("Generating...", dont_reset=True)

        app_support.set_seed(seed)

        results = []
        for prompt in prompts:
            try:
                audio_arrays: list[np.ndarray] = self._model.generate(
                    text=prompt,
                    language=language,
                    speed=speed,
                    generation_config=generation_config,
                )
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.INFO.default_output_sample_rate))

            except Exception as e:
                return make_error_string(e)

        return results

    @staticmethod
    def _print_params(d: dict, config: OmniVoiceGenerationConfig):
        from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
        d["num_step"] = config.num_step
        d["guidance_scale"] = config.guidance_scale
        TtsBaseModel.print_params(d, keys_blacklist=["generation_config"])
