import random
from collections.abc import Iterable
from pathlib import Path


import numpy as np
import torch

from omnivoice import OmniVoice  # type: ignore
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig  # type: ignore
from omnivoice.models.omnivoice import VoiceClonePrompt  # type: ignore

from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types import DeviceType, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_models.lora_switch import BaseWeightCache, LoraSwitcher, SafetensorsBaseWeights
from tts_audiobook_tool.util import *
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


def _release_cached_vram_if_large() -> None:
    """
    Releases PyTorch's cached-but-unused CUDA memory, but only when it is
    significant (more than 1 GiB above what live tensors actually use).

    The caching allocator keeps freed blocks reserved for reuse and never
    returns them to the driver on its own, so any transient peak would
    otherwise stay reserved for the rest of the session. That cache is not
    usable by other runtimes sharing the GPU (faster-whisper/CTranslate2,
    ONNX Runtime), which can push the total into shared system memory.

    Calling empty_cache() after every segment would force the allocator to
    re-request memory from the driver each time, so it is skipped when the
    cache is small.
    """
    if not torch.cuda.is_available():
        return
    if torch.cuda.memory_reserved() > torch.cuda.memory_allocated() + 1024**3:
        torch.cuda.empty_cache()


class OmniVoiceModel(OmniVoiceBaseModel):
    """
    App wrapper for `omnivoice.OmniVoice`
    Supports Voice Cloning, Voice Design and Auto Voice modes.

    Per-voice LoRAs (`Voice.loras`, PEFT format, `omnivoice_lora`) are merged into the weights and
    switched in place before each call that needs a different one (`lora_switch.LoraSwitcher`).
    A voice's LoRA strength scales the adapter's whole change, its embedding/head replacements
    included (`omnivoice_lora`).
    """

    # Effectively disable OmniVoice's own long-text chunker
    AUDIO_CHUNK_THRESHOLD_SECONDS = 999.0

    # Voice clone prompts are small (C x T int codes) and CPU-retained, so
    # several voices can be kept at once.
    RETAINS_MULTIPLE_VOICE_CLONES = True

    def __init__(self, model_target: ModelTarget, device: DeviceType):

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

        # Resolve once so base-weight restores use the exact checkpoint that was loaded.
        model_dir = Path(model_target.resolve_local_dir())
        self._model: OmniVoice = OmniVoice.from_pretrained(
            str(model_dir),
            device_map=device_map,
            dtype=dtype,
        )

        self._loras: LoraSwitcher | None = self._make_lora_switcher(self._model, model_dir)

    @staticmethod
    def _make_lora_switcher(model: torch.nn.Module, model_dir: Path) -> LoraSwitcher:
        from tts_audiobook_tool.tts_models.omnivoice_lora import prepare_lora

        def planner(lora: VoiceLora):
            return prepare_lora(model, lora.target, lora.strength)

        # OmniVoice's parameter names match its checkpoint keys. The source casts
        # fp32 checkpoint values to the loaded dtype (fp16 on CUDA) before restoring.
        return LoraSwitcher(model, BaseWeightCache(SafetensorsBaseWeights(str(model_dir))), planner)

    def prepare_voice_loras(self, project: Project, voice_indices: Iterable[int]) -> list[str]:
        if self._loras is None:
            return []
        problems = []
        voices = project.voice_references
        for index in sorted(set(voice_indices)):
            if not 0 <= index < len(voices):
                continue
            lora = ProjectVoiceUtil.active_voice_lora(project, voices[index])
            if lora is None:
                continue
            error = self._loras.prepare(lora)
            if error:
                problems.append(f"Voice {index + 1} LoRA ({lora.target.display_name()}): {error}")
        return problems

    def validate_voice_lora(self, lora: VoiceLora) -> str:
        if self._loras is None:
            return "Logic error - model not initialized"
        # Menu probes have their own preparation scope, never a stale run's cache.
        self._loras.end_run()
        try:
            return self._loras.prepare(lora)
        finally:
            self._loras.end_run()

    def end_generation_run(self) -> None:
        if self._loras is not None:
            self._loras.end_run()

    def release_inference_memory(self) -> None:
        # Chat only ever uses voice 1, so it switches LoRA at most once: drop the base copy, but keep
        # the plan, so later chat lines don't re-resolve and re-hash the adapter (as Breeze).
        if self._loras is not None:
            self._loras.release_base()
        _release_cached_vram_if_large()

    def kill(self) -> None:
        self.end_generation_run()
        self._loras = None
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

        model_type = TtsModelType.require_by_id("omnivoice_local")
        voice_file_name, ref_text = ProjectVoiceUtil.current_voice_reference_pair(
            project, model_type, voice_selection_index
        )
        voice_path = ProjectVoiceUtil.resolve_model_voice_file_path(project, voice_file_name) if voice_file_name else ""
        # Instructions belong to the selected voice, keyed by model id.
        voice = ProjectVoiceUtil.current_voice(project, model_type, voice_selection_index)
        instruct = voice.instructions.get(model_type.id, "") if voice is not None else ""
        lora = ProjectVoiceUtil.active_voice_lora(project, voice)
        if self._loras is None:
            return "Logic error - model not initialized"
        error = self._loras.activate(lora)
        if error:
            return f"Couldn't apply voice {voice_selection_index + 1}'s LoRA: {error}"
        lora_display = self._loras.active_display if lora is not None else ""
        lora_strength = lora.strength if lora is not None else 1.0

        cfg      = self.LOCAL.resolve_float(project, "cfg")
        speed    = self.LOCAL.resolve_float(project, "speed")
        steps    = self.LOCAL.resolve_int(project, "num_step")
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

        # OmniVoice id, or None (language-agnostic mode) when empty or unknown
        language = self.resolve_language(project).value

        if has_voice:
            return self._generate_voice_clone(
                prompts=prompts,
                language=language,
                voice_path=voice_path,
                ref_text=ref_text,
                instruct=instruct,
                speed=speed,
                seed=seed,
                generation_config=generation_config,
                print_params=print_params,
                lora=lora_display,
                lora_strength=lora_strength,
            )
        elif has_instruct:
            return self._generate_voice_design(
                prompts=prompts,
                language=language,
                instruct=instruct,
                speed=speed,
                seed=seed,
                generation_config=generation_config,
                print_params=print_params,
                lora=lora_display,
                lora_strength=lora_strength,
            )
        else:
            return self._generate_auto_voice(
                prompts=prompts,
                language=language,
                speed=speed,
                seed=seed,
                generation_config=generation_config,
                print_params=print_params,
                lora=lora_display,
                lora_strength=lora_strength,
            )

    def _create_voice_clone(self, source_path: str, transcript: str) -> VoiceClonePrompt:
        """
        Creates the library's reusable prompt from the reference audio and
        returns a CPU clone of it.

        Raising here aborts the generation with an error string (handled by
        the caller) and nothing gets cached.
        """
        # The library's create_voice_clone_prompt() is not decorated with
        # inference_mode (unlike its generate()), and the Higgs audio
        # tokenizer's acoustic encoder runs with autograd enabled. Called
        # directly, the encoder saves every intermediate activation for a
        # backward pass, peaking at ~300 MiB extra per second of reference
        # audio (~4.5 GiB for a 16 s reference) instead of ~30 MiB/s. The
        # graph is freed when encoding returns, but PyTorch's caching
        # allocator keeps the peak reserved for the rest of the session.
        # Repro: testx/omnivoice_clone_prompt_vram.py
        with torch.inference_mode():
            prompt = self._model.create_voice_clone_prompt(
                ref_audio=source_path,
                ref_text=transcript or None,
            )
        # Clone outside inference_mode so the cached tokens are ordinary
        # tensors rather than inference tensors, which would be restricted
        # anywhere outside inference_mode.
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
            language: str | None,
            print_params: bool = False,
            lora: str = "",
            lora_strength: float = 1.0,
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

        # Print something here b/c model library code does not
        printt(f"{COL_DIM_ITALICS}Generating...")

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
                _release_cached_vram_if_large()

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
            language: str | None,
            print_params: bool = False,
            lora: str = "",
            lora_strength: float = 1.0,
    ) -> list[Sound] | str:

        if print_params:
            self._print_params(locals(), generation_config)

        # Print something here b/c model library code does not
        printt(f"{COL_DIM_ITALICS}Generating...")

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
                _release_cached_vram_if_large()

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
            language: str | None,
            print_params: bool = False,
            lora: str = "",
            lora_strength: float = 1.0,
    ) -> list[Sound] | str:

        if print_params:
            self._print_params(locals(), generation_config)

        # Print something here b/c model library code does not
        printt(f"{COL_DIM_ITALICS}Generating...")

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
                _release_cached_vram_if_large()

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
        # The LoRA (and its strength) only when the voice has one
        blacklist = ["generation_config"] if d.get("lora") else ["generation_config", "lora", "lora_strength"]
        TtsBaseModel.print_params(d, keys_blacklist=blacklist)
