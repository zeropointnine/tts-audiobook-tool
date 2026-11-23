import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from typing import Callable

from tts_audiobook_tool import app_support
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference # type: ignore
from vibevoice.modular.streamer import AudioStreamer # type: ignore
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor # type: ignore
from tts_audiobook_tool.app_types import DeviceType, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.tts_models.lora_switch import BaseWeightCache, LoraSwitcher, SafetensorsBaseWeights
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


class CallbackAudioStreamer(AudioStreamer):
    def __init__(
            self,
            batch_size: int,
            on_chunk: Callable[[int, torch.Tensor], None],
            on_end: Callable[[], None] | None = None,
    ):
        super().__init__(batch_size=batch_size)
        self.on_chunk = on_chunk
        self.on_end = on_end
        self.did_notify_end = False

    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        super().put(audio_chunks, sample_indices)
        for i, sample_idx in enumerate(sample_indices):
            self.on_chunk(int(sample_idx.item()), audio_chunks[i].detach().cpu())

    def end(self, sample_indices=None):
        super().end(sample_indices)
        if sample_indices is None and self.on_end is not None and not self.did_notify_end:
            self.did_notify_end = True
            self.on_end()


class VibeVoiceModel(VibeVoiceBaseModel):
    """
    VibeVoice TTS inference logic
    Mostly copy-pasted from: VibeVoice/demo/inference_from_file.py

    Per-voice LoRAs (`Voice.loras`, PEFT language model adapters, `vibevoice_lora`) are merged into
    the weights and switched in place before each call that needs a different one
    (`lora_switch.LoraSwitcher`). Generation speed is unaffected; a switch costs ~0.3 s for 1.5B
    (its adapters touch ~2.4 GB of weights; a run's first restore, which builds the pinned base copy,
    ~2 s), several seconds for 7B (~12 GB).
    """
    def __init__(
            self,
            device: DeviceType,
            model_target: ModelTarget | None = None,
            max_new_tokens: int | None = None,
    ):

        self._device_type = device
        device_value = device.value
        if model_target is None:
            model_target = VibeVoiceBaseModel.DEFAULT_TARGET
        self.max_new_tokens = max_new_tokens
        self.audio_streamer: AudioStreamer | None = None
        self.stream_chunk_callback: StreamChunkCallback | None = None
        self.stream_end_callback: StreamEndCallback | None = None

        # Resolve once so base-weight restores use the exact checkpoint that was loaded
        model_dir = Path(model_target.resolve_local_dir())

        self.processor = VibeVoiceProcessor.from_pretrained(str(model_dir))

        # Determine attention implementation type
        if device == DeviceType.CUDA:
            try:
                from flash_attn import flash_attn_func # type: ignore
                attn_implementation = "flash_attention_2"
            except ImportError as e:
                attn_implementation = "sdpa" # fyi, in my testing, "eager" is unusable
                # ... note, triton is still required though
        else:
            attn_implementation = "sdpa"
        printt(f"Attention implementation: {attn_implementation}")

        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map=device_value,
            attn_implementation=attn_implementation
        )
        self.model.eval()

        self._loras: LoraSwitcher | None = VibeVoiceModel._make_lora_switcher(self.model, model_dir)

    @staticmethod
    def _make_lora_switcher(model: torch.nn.Module, model_dir: Path) -> LoraSwitcher:
        from tts_audiobook_tool.tts_models.vibevoice_lora import prepare_lora

        def planner(lora: VoiceLora):
            return prepare_lora(model, lora.target, lora.strength)

        # VibeVoice's parameter names are its checkpoint keys, and checkpoint and model are both bf16
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
        # Menu probes have their own preparation scope, never a stale run's cache
        self._loras.end_run()
        try:
            return self._loras.prepare(lora)
        finally:
            self._loras.end_run()

    def end_generation_run(self) -> None:
        if self._loras is not None:
            self._loras.end_run()

    def release_inference_memory(self) -> None:
        # Chat only ever uses voice 1, so it switches LoRA at most once: drop the base copy,
        # but keep the plan, so later chat lines don't re-resolve and re-hash the adapter
        if self._loras is not None:
            self._loras.release_base()

    def kill(self) -> None:
        self.end_generation_run()
        self._loras = None
        if self.processor:
            self.processor.tokenizer = None
            self.processor.audio_processor = None
            self.processor.audio_normalizer = None

        self.stream_chunk_callback = None
        self.stream_end_callback = None
        self.audio_streamer = None
        self.clear_stream_state()
        self.processor = None
        self.model = None

    def clear_stream_state(self) -> None:
        super().clear_stream_state()
        self.audio_streamer = None

    @classmethod
    def massage_for_inference(cls, text: str) -> str:
        text = super().massage_for_inference(text)
        # Required speaker tag
        text = f"{SPEAKER_TAG}{text}"
        return text

    def on_audio_chunk_received(self, sample_idx: int, chunk: torch.Tensor) -> None:
        if chunk.dtype == torch.bfloat16:
            chunk = chunk.to(torch.float32)

        chunk_data = chunk.numpy().astype(np.float32, copy=False).reshape(-1)

        if self.stream_chunk_callback is not None:
            self.stream_chunk_callback(chunk_data)

    def on_stream_end(self) -> None:
        if self.stream_end_callback is not None:
            self.stream_end_callback()

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_selection_index: int = 0,
            print_params: bool = False,
            max_random_seed: int = -1,
        ) -> list[Sound] | str:

        model_type = TtsModelType.require_by_id("vibevoice_local")
        voice_file_name = ProjectVoiceUtil.current_voice_value(project, model_type, voice_selection_index)
        if voice_file_name:
            voice_path = ProjectVoiceUtil.resolve_model_voice_file_path(project, voice_file_name)
        else:
            voice_path = ""

        # The voice's LoRA, or none: switched in place when it differs from the merged one
        voice = ProjectVoiceUtil.current_voice(project, model_type, voice_selection_index)
        lora = ProjectVoiceUtil.active_voice_lora(project, voice)
        if self._loras is None:
            return "Logic error - model not initialized"
        error = self._loras.activate(lora)
        if error:
            return f"Couldn't apply voice {voice_selection_index + 1}'s LoRA: {error}"
        lora_display = f"{self._loras.active_display} (strength {lora.strength:g})" if lora is not None else ""

        cfg_scale = VibeVoiceBaseModel.LOCAL.resolve_float(project, "cfg")
        num_steps = VibeVoiceBaseModel.LOCAL.resolve_int(project, "steps")

        seed = -1 if force_random_seed else project.get_model_setting('vibevoice_local', 'seed')
        if seed <= -1:
            seed = random.randrange(0, get_random_seed_max(SEED_MAX - 1, max_random_seed) + 1)

        result = self.generate(
            texts=prompts,
            voice_path=voice_path,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            on_stream_chunk=on_stream_chunk,
            on_stream_end=on_stream_end,
            print_params=print_params,
            lora=lora_display,
        )
        return result

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            seed: int,
            cfg_scale: float,
            num_steps: int,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            print_params: bool = False,
            lora: str = "",
    ) -> list[Sound] | str:
        """
        Returns list[Sound] or error string.
        `lora` only names the merged LoRA for `print_params` (the caller activates it).
        All values must be concrete (ie, resolved defaults, randomized seed).

        FYI: Couldn't pass pre-loaded sound data without inference issues for some reason,
        but in practice overhead is negligible
        """
        if self.model is None or self.processor is None:
            return "model or processor is not initialized" # logic error

        if print_params:
            from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
            params = locals().copy()
            if not lora:
                del params["lora"]
            TtsBaseModel.print_params(params)

        app_support.set_seed(seed)

        try:
            self.model.set_ddpm_inference_steps(num_steps) # type: ignore
            self.stream_chunk_callback = on_stream_chunk
            self.stream_end_callback = on_stream_end

            # Ensure text is a list
            texts = texts if isinstance(texts, list) else [texts]

            voice_samples = None if not voice_path else [ [ voice_path ] for _ in texts ]

            inputs = self.processor(
                text=texts,
                voice_samples=voice_samples,  # type: ignore
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            self.audio_streamer = None
            if on_stream_chunk is not None or on_stream_end is not None:
                self.audio_streamer = CallbackAudioStreamer(
                    batch_size=len(texts),
                    on_chunk=self.on_audio_chunk_received,
                    on_end=self.on_stream_end,
                )

            # Generate audio
            outputs = self.model.generate(
                **inputs, # type: ignore
                max_new_tokens=self.max_new_tokens,
                max_length_times=VibeVoiceBaseModel.MAX_LENGTH_TIMES,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                audio_streamer=self.audio_streamer,
                # generation_config={'do_sample': False, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 0},
                generation_config={'do_sample': False}, # type: ignore
                verbose=True,
            )
        except Exception as e:
            return make_error_string(e)

        if not outputs.speech_outputs: # type: ignore
            return "No audio output"

        sounds = []
        for i, data in enumerate(outputs.speech_outputs): # type: ignore
            if data is None:
                return f"No audio output for item {i}"

            # Is bfloat16
            tensor_data = data[0]
            if tensor_data.dtype == torch.bfloat16:
                tensor_data = tensor_data.to(torch.float32)

            ndarray_data = tensor_data.cpu().numpy()
            sound = Sound(ndarray_data, self.INFO.default_output_sample_rate)
            sounds.append(sound)

        return sounds

SPEAKER_TAG = "Speaker 1: "
