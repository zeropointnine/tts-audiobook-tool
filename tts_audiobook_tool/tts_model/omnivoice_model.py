import numpy as np
import torch

from omnivoice import OmniVoice  # type: ignore

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.util import *


class OmniVoiceModel(OmniVoiceBaseModel):
    """
    App wrapper for `omnivoice.OmniVoice`
    Supports Voice Cloning, Voice Design and Auto Voice modes.
    """

    def __init__(self, model_target: str, device: str, dtype_str: str = "float16"):

        self._model_target = model_target
        self._device = device
        self._voice_info: tuple[str, str] | None = None  # cache (path, transcript)

        if device == "cuda":
            device_map = "cuda:0"
        else:
            device_map = device  # "mps" or "cpu"

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.float16)
        if device in ("cpu", "mps"):
            dtype = torch.float32  # CPU/MPS doesn't support fp16

        self._model: OmniVoice = OmniVoice.from_pretrained(
            model_target,
            device_map=device_map,
            dtype=dtype,
        )

    def kill(self) -> None:
        self._model = None  # type: ignore
        self._voice_info = None

    # ── Main interface ────────────────────────────────────────────────

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False
    ) -> list[Sound] | str:

        voice_path = os.path.join(project.dir_path, project.omnivoice_voice_file_name) \
                     if project.omnivoice_voice_file_name else ""
        ref_text = project.omnivoice_voice_transcript
        instruct = project.omnivoice_instruct
        speed    = project.omnivoice_speed if project.omnivoice_speed != -1 else 1.0
        num_step = project.omnivoice_num_step if project.omnivoice_num_step != -1 else 32
        seed     = -1 if force_random_seed else project.omnivoice_seed

        has_voice    = bool(voice_path and os.path.isfile(voice_path))
        has_instruct = bool(instruct)

        if has_voice:
            return self._generate_voice_clone(
                prompts=prompts,
                voice_path=voice_path,
                ref_text=ref_text,
                instruct=instruct,
                speed=speed,
                num_step=num_step,
                seed=seed,
            )
        elif has_instruct:
            return self._generate_voice_design(
                prompts=prompts,
                instruct=instruct,
                speed=speed,
                num_step=num_step,
                seed=seed,
            )
        else:
            return self._generate_auto_voice(
                prompts=prompts,
                speed=speed,
                num_step=num_step,
                seed=seed,
            )

    # ── Generation modes ──────────────────────────────────────────────────

    def _generate_voice_clone(
            self,
            prompts: list[str],
            voice_path: str,
            ref_text: str,
            instruct: str,
            speed: float,
            num_step: int,
            seed: int,
    ) -> list[Sound] | str:

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        printt("Generating...", dont_reset=True)

        results = []
        for prompt in prompts:
            try:
                kw: dict = dict(text=prompt, ref_audio=voice_path, speed=speed, num_step=num_step,)
                if ref_text:
                    kw["ref_text"] = ref_text   # omitted → OmniVoice already uses Whisper internally
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
            speed: float,
            num_step: int,
            seed: int,
    ) -> list[Sound] | str:

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        printt("Generating...", dont_reset=True)

        results = []
        for prompt in prompts:
            try:
                audio_arrays: list[np.ndarray] = self._model.generate(
                    text=prompt,
                    instruct=instruct,
                    speed=speed,
                    num_step=num_step,
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
            speed: float,
            num_step: int,
            seed: int,
    ) -> list[Sound] | str:

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        printt("Generating...", dont_reset=True)

        results = []
        for prompt in prompts:
            try:
                audio_arrays: list[np.ndarray] = self._model.generate(
                    text=prompt,
                    speed=speed,
                    num_step=num_step,
                )
                audio = audio_arrays[0].astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                results.append(Sound(audio, self.SAMPLE_RATE))

            except Exception as e:
                return make_error_string(e)

        return results
