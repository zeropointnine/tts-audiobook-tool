import numpy as np
import torch

from omnivoice import OmniVoice  # type: ignore
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig  # type: ignore
from omnivoice.models.omnivoice import VoiceClonePrompt  # type: ignore

from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.util import *


class OmniVoiceModel(OmniVoiceBaseModel):
    """
    App wrapper for `omnivoice.OmniVoice`
    Supports Voice Cloning, Voice Design and Auto Voice modes.
    """

    # Effectively disable OmniVoice's own long-text chunker
    AUDIO_CHUNK_THRESHOLD_SECONDS = 999.0

    def __init__(self, model_target: str, device: str):

        self._model_target = model_target
        self._device = device
        self._voice_info: tuple[str, str] | None = None
        self._voice_clone_prompt: VoiceClonePrompt | None = None

        if device == "cuda":
            device_map = "cuda:0"
        else:
            device_map = device  # "mps" or "cpu"

        if device == "cuda":
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

        self._model = None  # type: ignore
        self._voice_info = None
        self._voice_clone_prompt = None

    # ── Main interface ────────────────────────────────────────────────

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None
    ) -> list[Sound] | str:

        voice_path = os.path.join(project.dir_path, project.omnivoice_voice_file_name) \
                     if project.omnivoice_voice_file_name else ""
        ref_text = project.omnivoice_voice_transcript
        instruct = project.omnivoice_instruct
        cfg      = project.omnivoice_cfg if project.omnivoice_cfg != -1 else self.CFG_DEFAULT
        speed    = project.omnivoice_speed if project.omnivoice_speed != -1 else self.DEFAULT_SPEED
        steps    = project.omnivoice_num_step if project.omnivoice_num_step != -1 else self.DEFAULT_STEPS
        seed     = -1 if force_random_seed else project.omnivoice_seed

        has_voice    = bool(voice_path and os.path.isfile(voice_path))
        has_instruct = bool(instruct)

        if has_voice:
            return self._generate_voice_clone(
                prompts=prompts,
                voice_path=voice_path,
                ref_text=ref_text,
                instruct=instruct,
                cfg=cfg,
                speed=speed,
                steps=steps,
                seed=seed,
            )
        elif has_instruct:
            return self._generate_voice_design(
                prompts=prompts,
                instruct=instruct,
                cfg=cfg,
                speed=speed,
                steps=steps,
                seed=seed,
            )
        else:
            return self._generate_auto_voice(
                prompts=prompts,
                cfg=cfg,
                speed=speed,
                steps=steps,
                seed=seed,
            )

    # ── Generation modes ──────────────────────────────────────────────────

    def _generate_voice_clone(
            self,
            prompts: list[str],
            voice_path: str,
            ref_text: str,
            instruct: str,
            cfg: float,
            speed: float,
            steps: int,
            seed: int,
    ) -> list[Sound] | str:

        voice_info = (voice_path, ref_text)
        generation_config = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=cfg,
            audio_chunk_threshold=self.AUDIO_CHUNK_THRESHOLD_SECONDS,
        )

        if not self._voice_clone_prompt or self._voice_info != voice_info:
            try:
                self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                    ref_audio=voice_path,
                    ref_text=ref_text or None,
                )
            except Exception as e:
                return f"Couldn't create voice clone for {voice_path} - {make_error_string(e)}"

            self._voice_info = voice_info

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        printt("Generating...", dont_reset=True)

        results = []
        for prompt in prompts:
            try:
                kw: dict = dict(
                    text=prompt,
                    voice_clone_prompt=self._voice_clone_prompt,
                    speed=speed,
                    generation_config=generation_config,
                )
                if instruct:
                    kw["instruct"] = instruct   # cloning + combined styles

                audio_arrays: list[np.ndarray] = self._model.generate(**kw)
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.SAMPLE_RATE))

            except Exception as e:
                return make_error_string(e)

        return results

    def _generate_voice_design(
            self,
            prompts: list[str],
            instruct: str,
            cfg: float,
            speed: float,
            steps: int,
            seed: int,
    ) -> list[Sound] | str:

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        generation_config = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=cfg,
            audio_chunk_threshold=self.AUDIO_CHUNK_THRESHOLD_SECONDS,
        )

        printt("Generating...", dont_reset=True)

        results = []
        for prompt in prompts:
            try:
                audio_arrays: list[np.ndarray] = self._model.generate(
                    text=prompt,
                    instruct=instruct,
                    speed=speed,
                    generation_config=generation_config,
                )
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.SAMPLE_RATE))

            except Exception as e:
                return make_error_string(e)

        return results

    def _generate_auto_voice(
            self,
            prompts: list[str],
            cfg: float,
            speed: float,
            steps: int,
            seed: int,
    ) -> list[Sound] | str:

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        generation_config = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=cfg,
            audio_chunk_threshold=self.AUDIO_CHUNK_THRESHOLD_SECONDS,
        )

        printt("Generating...", dont_reset=True)

        results = []
        for prompt in prompts:
            try:
                audio_arrays: list[np.ndarray] = self._model.generate(
                    text=prompt,
                    speed=speed,
                    generation_config=generation_config,
                )
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.SAMPLE_RATE))

            except Exception as e:
                return make_error_string(e)

        return results
