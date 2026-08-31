from __future__ import annotations

import os
import random

import numpy as np
import torch
from dots_tts.runtime import DotsTtsRuntime  # type: ignore
from dots_tts.utils.util import seed_everything  # type: ignore

from tts_audiobook_tool.app_types import (
    DeviceType,
    Sound,
    StreamChunkCallback,
    StreamEndCallback,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.dots_base_model import (
    DotsBaseModel,
    DotsCompileMode,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_error_string, printt


class DotsModel(DotsBaseModel):

    # FYI, no tool-side voice clone cache, and so not opting in to
    # RETAINS_MULTIPLE_VOICE_CLONES: the dots runtime caches the derived voice
    # intermediates (speaker embedding, prompt latents) internally, in a
    # 256-entry LRU keyed by the reference audio's content hash. Re-selecting a
    # voice - including round-robin switching between several voices - therefore
    # hits the library cache already; the per-call CPU-side work it repeats
    # (librosa load/trim/resample plus the cache-key hash) is negligible next
    # to generation.

    def __init__(
        self,
        model_target: str,
        device: DeviceType,
        compile_enabled: bool = DotsCompileMode.default().enabled,
    ):
        if device == DeviceType.CPU and torch.cuda.is_available():
            raise RuntimeError(
                "dots.tts cannot force CPU inference while CUDA is visible; "
                "the dots runtime currently selects its device automatically"
            )
        if device not in {DeviceType.CUDA, DeviceType.CPU}:
            raise ValueError(f"Unsupported dots.tts device: {device.value}")

        self._device_type = device
        self._model_target = model_target
        if device == DeviceType.CUDA:
            # Let Inductor use TF32 tensor cores for faster float32 matrix multiplies.
            torch.set_float32_matmul_precision("high")
        precision = "bfloat16" if device == DeviceType.CUDA else "float32"
        optimize = compile_enabled and device == DeviceType.CUDA
        runtime = DotsTtsRuntime.from_pretrained(
            model_target,
            precision=precision,
            optimize=optimize,
            max_generate_length=DotsBaseModel.MAX_GENERATE_LENGTH,
        )
        self._runtime: DotsTtsRuntime | None = runtime
        if runtime.sample_rate != self.INFO.sample_rate:
            raise ValueError(
                f"Unexpected dots.tts output sample rate: {runtime.sample_rate} "
                f"(expected {self.INFO.sample_rate})"
            )

    @property
    def model_target(self) -> str:
        return self._model_target

    def kill(self) -> None:
        self._runtime = None

    def generate_using_project(
        self,
        project: Project,
        prompts: list[str],
        force_random_seed: bool = False,
        on_stream_chunk: StreamChunkCallback | None = None,
        on_stream_end: StreamEndCallback | None = None,
        voice_selection_index: int = 0,
    ) -> list[Sound] | str:
        voice_file_name, voice_transcript = ProjectVoiceUtil.current_voice_reference_pair(
            project, TtsModelType.DOTS, voice_selection_index
        )
        voice_path = (
            os.path.join(project.dir_path, voice_file_name) if voice_file_name else None
        )
        seed = -1 if force_random_seed else project.dots_seed
        speaker_scale = (
            self.SPEAKER_SCALE_DEFAULT
            if project.dots_speaker_scale == -1
            else project.dots_speaker_scale
        )
        num_steps = self.resolve_num_steps(project)
        guidance_scale = (
            self.GUIDANCE_SCALE_DEFAULT
            if project.dots_guidance_scale == -1
            else project.dots_guidance_scale
        )
        return self.generate(
            prompts=prompts,
            voice_path=voice_path,
            voice_transcript=voice_transcript,
            language=project.language_code.strip() or None,
            speaker_scale=speaker_scale,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            on_stream_chunk=on_stream_chunk,
            on_stream_end=on_stream_end,
        )

    def generate(
        self,
        prompts: list[str],
        voice_path: str | None,
        voice_transcript: str,
        language: str | None,
        speaker_scale: float,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        on_stream_chunk: StreamChunkCallback | None = None,
        on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:
        runtime = self._runtime
        if runtime is None:
            return "Logic error - dots.tts model not initialized"

        if seed == -1:
            seed = random.randrange(0, self.SEED_MAX)
        seed_everything(seed)

        sampling_locked = runtime.model.config.sampling is not None
        # Meanflow artifacts have no CFG branch (guidance is distilled into
        # the model), so a user guidance value would be silently discarded.
        meanflow = (
            runtime.model.config.meanflow is not None
            and runtime.model.config.meanflow.enabled
        )
        resolved_num_steps = None if sampling_locked else num_steps
        resolved_guidance_scale = (
            None if (sampling_locked or meanflow) else guidance_scale
        )
        is_streaming = on_stream_chunk is not None or on_stream_end is not None

        printt("Generating...", dont_reset=True)
        sounds: list[Sound] = []
        try:
            for prompt in prompts:
                if is_streaming:
                    audio_chunks: list[np.ndarray] = []
                    stream = runtime.generate_stream(
                        text=prompt,
                        prompt_audio_path=voice_path,
                        prompt_text=voice_transcript or None,
                        template_name=None,
                        language=language,
                        speaker_scale=speaker_scale,
                        ode_method=None,
                        num_steps=resolved_num_steps,
                        guidance_scale=resolved_guidance_scale,
                        normalize_text=False,
                    )
                    for chunk in stream:
                        chunk_audio = (
                            chunk.detach()
                            .to(dtype=torch.float32)
                            .cpu()
                            .reshape(-1)
                            .numpy()
                        )
                        if chunk_audio.size == 0:
                            continue
                        audio_chunks.append(chunk_audio)
                        if on_stream_chunk is not None:
                            on_stream_chunk(chunk_audio)
                    if not audio_chunks:
                        raise ValueError("dots.tts returned empty audio")
                    audio = np.concatenate(audio_chunks)
                    sample_rate = runtime.sample_rate
                else:
                    result = runtime.generate(
                        text=prompt,
                        prompt_audio_path=voice_path,
                        prompt_text=voice_transcript or None,
                        template_name=None,
                        language=language,
                        speaker_scale=speaker_scale,
                        ode_method=None,
                        num_steps=resolved_num_steps,
                        guidance_scale=resolved_guidance_scale,
                        normalize_text=False,
                    )
                    sample_rate = int(result["sample_rate"])
                    audio = (
                        result["audio"]
                        .detach()
                        .to(dtype=torch.float32)
                        .cpu()
                        .reshape(-1)
                        .numpy()
                    )

                if sample_rate != self.INFO.sample_rate:
                    raise ValueError(
                        f"Unexpected dots.tts output sample rate: {sample_rate} "
                        f"(expected {self.INFO.sample_rate})"
                    )
                if audio.size == 0:
                    raise ValueError("dots.tts returned empty audio")
                sounds.append(Sound(audio, sample_rate))

            if is_streaming and on_stream_end is not None:
                on_stream_end()
        except Exception as e:
            return make_error_string(e)
        return sounds
