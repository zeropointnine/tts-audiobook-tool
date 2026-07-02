import os
import random
import time
from importlib import util as importlib_util

import numpy as np
import torch
import huggingface_hub

from tts_audiobook_tool import app_support, target_util
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import SEED_MAX
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.moss_base_model import MossArchType, MossConfigs, MossBaseModel
from tts_audiobook_tool.util import *

from transformers import AutoModel, AutoProcessor  # type: ignore


class MossModel(MossBaseModel):

    def __init__(self, device: str, model_target: str = MossConfigs.get_default_repo_id()):

        self.model_target = model_target
        self.device = device or "cpu"
        self.dtype = self.resolve_dtype()
        self.cached_voice_path = ""
        self.cached_voice_stat: tuple[int, int] | None = None
        self.cached_voice_codes: torch.Tensor | None = None
        self.cached_continuation_history: list[tuple[str, torch.Tensor]] = []
        self.audio_tokenizer_is_on_device = False

        if self.device == "cuda":
            # Recommended by MOSS-TTS upstream for the remote-code generation path.
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        attn_implementation = self.resolve_attn_implementation()
        printt(f"Attention implementation: {attn_implementation}")

        if target_util.is_hf_repo_id_syntax(model_target):
            
            # TODO: Should not derive revision by repo id. Should pass revision into ctor.
            preset = MossConfigs.get_preset_by_target(model_target)
            revision = preset.value.revision if preset else None
            if revision:
                printt(f"Using pinned MOSS-TTS revision: {revision[:12]}")
            
            local_path = huggingface_hub.snapshot_download(
                repo_id=model_target,
                revision=revision,
                cache_dir=huggingface_hub.constants.HF_HUB_CACHE,
                local_files_only=False,
            )
        else:
            local_path = model_target

        # Due to transformers bug, can't simply use hf repo id here bc on Windows, 
        # it mangles the repo id (uses backslash instead of forward slash)

        self.processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=True,
        )
        if hasattr(self.processor, "audio_tokenizer"):
            self.processor.audio_tokenizer.eval()

        # Rem, this can raise exception (eg OOM), which should handled properly by instantiator
        self.model = self.load_model(local_path, attn_implementation)
        self.model.eval()

        printt(f"MOSS loaded arch type: {self.get_loaded_arch_type().value}")

    def resolve_dtype(self) -> torch.dtype:
        if self.device != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def resolve_attn_implementation(self) -> str:
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if (
                device.type == "cuda"
                and importlib_util.find_spec("flash_attn") is not None
                and self.dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                return "flash_attention_2"
        if device.type == "cuda":
            return "sdpa"
        return "eager"

    def load_model(self, local_path: str, attn_implementation: str):
        printt(f"Loading MOSS-TTS with attention implementation: {attn_implementation}")
        return AutoModel.from_pretrained(
            local_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=self.dtype,
        ).to(self.device)

    def get_loaded_arch_type(self) -> MossArchType:

        if self.model is None:
            raise RuntimeError("Model is not initialized")

        # To infer if model is of the 'Local' type, must test for layer info or local transformer
        config = getattr(self.model, "config", None)
        local_num_layers = getattr(config, "local_num_layers", None)
        has_local_transformer = hasattr(self.model, "local_transformer")
        
        if has_local_transformer or local_num_layers is not None:
            return MossArchType.LOCAL
        if type(self.model).__name__ == "MossTTSDelayModel":
            return MossArchType.DELAY
        return MossArchType.UNKNOWN

    def get_memory_usage(self) -> str:
        parts = []
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            parts.append(f"torch_allocated={allocated_gb:.2f}GB")
            parts.append(f"torch_reserved={reserved_gb:.2f}GB")
        nv_vram = app_memory.get_nv_vram()
        if nv_vram:
            used, total = nv_vram
            parts.append(f"nv_used={used / (1024 ** 3):.2f}GB")
            parts.append(f"nv_total={total / (1024 ** 3):.2f}GB")
        return ", ".join(parts) if parts else "unavailable"

    def prepare_audio_tokenizer(self, label: str) -> None:
        if self.processor is None:
            raise RuntimeError("Processor is not initialized")
        if not hasattr(self.processor, "audio_tokenizer"):
            return
        if self.audio_tokenizer_is_on_device:
            return
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)
        self.processor.audio_tokenizer.eval()
        self.audio_tokenizer_is_on_device = True

    @staticmethod
    def build_continuation_text(previous_text: str, prompt: str) -> str:
        parts = [part.strip() for part in [previous_text, prompt] if part.strip()]
        return " ".join(parts)

    @staticmethod
    def get_prompt_log_preview(prompt: str) -> str:
        return prompt.replace("\n", " ")[:50]

    def kill(self) -> None:
        if self.processor:
            if hasattr(self.processor, "audio_tokenizer"):
                self.processor.audio_tokenizer = None
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model.cpu()
        self.processor = None
        self.model = None
        self.cached_voice_path = ""
        self.cached_voice_stat = None
        self.cached_voice_codes = None
        self.clear_continuation()

    def generate_outputs(
            self,
            conversations,
            processor_mode: str,
            temperature: float,
            audio_top_p: float,
            audio_top_k: int,
    ):
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor is not initialized")

        batch = self.processor(conversations, mode=processor_mode)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.inference_mode():

            # Regarding hyperparams:

            # We are only setting audio_temperature (not also text_temperature),
            # matching project's demo behavior:
            # https://github.com/OpenMOSS/MOSS-TTS/blob/main/clis/moss_tts_app.py

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MossBaseModel.MAX_NEW_TOKENS,
                audio_temperature=temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
            )
            return outputs

    def decode_outputs_to_sounds_and_audio(self, outputs) -> tuple[list[Sound], list[torch.Tensor]] | str:
        if self.processor is None:
            return "Processor is not initialized"

        sounds = []
        audio_tensors = []
        sample_rate = getattr(self.processor.model_config, "sampling_rate", MossBaseModel.INFO.sample_rate)
        self.prepare_audio_tokenizer("decode")
        messages = self.processor.decode(outputs)
        for message in messages:
            if not message.audio_codes_list:
                return "No audio output"
            audio = message.audio_codes_list[0]
            audio_tensors.append(audio.detach().cpu() if isinstance(audio, torch.Tensor) else torch.as_tensor(audio))
            audio_np = audio.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
            sounds.append(Sound(audio_np, sample_rate))

        return sounds, audio_tensors

    def decode_outputs_to_sounds(self, outputs) -> list[Sound] | str:
        decoded = self.decode_outputs_to_sounds_and_audio(outputs)
        if isinstance(decoded, str):
            return decoded
        sounds, _ = decoded
        return sounds

    def clear_continuation(self) -> None:
        super().clear_continuation()
        self.cached_continuation_history.clear()

    def cache_continuation(
            self,
            prompt: str,
            audio: torch.Tensor,
            sample_rate: int,
            max_items: int,
    ) -> None:
        if max_items <= 0:
            self.clear_continuation()
            return
        if self.processor is None:
            raise RuntimeError("Processor is not initialized")

        self.prepare_audio_tokenizer("continuation wav encode")
        wav = audio.detach().to(torch.float32).cpu()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim > 2:
            wav = wav.reshape(-1).unsqueeze(0)

        encode_start = time.perf_counter()
        audio_codes = self.processor.encode_audios_from_wav([wav], sample_rate)[0]
        encode_duration = time.perf_counter() - encode_start

        self.cached_continuation_history.append((prompt.strip(), audio_codes.detach().cpu().clone()))
        while self.is_continuation_history_over_budget(max_items):
            self.cached_continuation_history.pop(0)

    def is_continuation_history_over_budget(self, max_items: int) -> bool:
        return (
            len(self.cached_continuation_history) > max_items
            or self.get_continuation_history_word_count() > ROLLING_CONTINUATION_MAX_WORDS
        )

    def get_continuation_history_word_count(self) -> int:
        return sum(
            self.get_word_count(prompt)
            for prompt, _ in self.cached_continuation_history
        )

    @staticmethod
    def get_word_count(text: str) -> int:
        return len(text.split())

    def build_rolling_continuation_reference(self) -> tuple[str, list[torch.Tensor]]:
        previous_text = " ".join(
            prompt.strip()
            for prompt, _ in self.cached_continuation_history
            if prompt.strip()
        )
        previous_audio_codes = [
            audio_codes.detach().cpu()
            for _, audio_codes in self.cached_continuation_history
        ]
        return previous_text, previous_audio_codes

    @staticmethod
    def build_audio_placeholder_content(num_items: int) -> str:
        return MOSS_AUDIO_PLACEHOLDER * num_items

    def get_voice_codes(self, voice_path: str) -> torch.Tensor | None:
        if not voice_path:
            return None
        if self.processor is None:
            raise RuntimeError("Processor is not initialized")

        normalized_voice_path = os.path.abspath(voice_path)
        stat = os.stat(normalized_voice_path)
        voice_stat = (stat.st_mtime_ns, stat.st_size)
        if (
                normalized_voice_path == self.cached_voice_path
                and voice_stat == self.cached_voice_stat
                and self.cached_voice_codes is not None
        ):
            return self.cached_voice_codes

        self.prepare_audio_tokenizer("voice path encode")
        voice_codes = self.processor.encode_audios_from_path([normalized_voice_path])[0]
        self.cached_voice_path = normalized_voice_path
        self.cached_voice_stat = voice_stat
        self.cached_voice_codes = voice_codes
        return voice_codes

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:

        if project.moss_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.moss_voice_file_name)
        else:
            voice_path = ""

        target = project.moss_target
        config = MossConfigs.get_by_target(target)

        temperature = project.moss_local_temperature if config == MossConfigs.LOCAL else project.moss_delay_temperature
        if temperature == -1:
            temperature = config.value.temperature_default

        audio_top_p = project.moss_local_top_p if config == MossConfigs.LOCAL else project.moss_delay_top_p
        if audio_top_p == -1:
            audio_top_p = config.value.audio_top_p_default

        audio_top_k = project.moss_local_top_k if config == MossConfigs.LOCAL else project.moss_delay_top_k
        if audio_top_k == -1:
            audio_top_k = config.value.audio_top_k_default

        seed = -1 if force_random_seed else project.moss_seed
        language = MossBaseModel.get_language_name(project.language_code) if project.language_code else ""

        return self.generate(
            prompts=prompts,
            voice_path=voice_path,
            rolling_continuation_max_segments=project.moss_rolling_cont,
            language=language,
            temperature=temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            seed=seed,
        )

    def generate(
            self,
            prompts: list[str],
            voice_path: str,
            rolling_continuation_max_segments: int,
            language: str,
            temperature: float,
            audio_top_p: float,
            audio_top_k: int,
            seed: int,
    ) -> list[Sound] | str:

        if self.model is None or self.processor is None:
            return "Model or processor is not initialized"

        voice_reference = os.path.abspath(voice_path) if voice_path else ""
        if voice_path and not os.path.exists(voice_reference):
            return "Missing voice reference audio"

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        app_support.set_seed(seed)

        try:

            if rolling_continuation_max_segments > 0:

                sounds = []

                for prompt_index, prompt in enumerate(prompts):
                    previous_text, previous_audio_codes = self.build_rolling_continuation_reference()

                    if previous_text and previous_audio_codes:
                        conversation_text = self.build_continuation_text(previous_text, prompt)
                        printt(f"{COL_DIM_ITALICS}Rolling continuation enabled, history length: {len(self.cached_continuation_history)}")
                        conversations = [[
                            self.processor.build_user_message(
                                text=conversation_text,
                                language=language or None,
                                reference=[voice_reference] if voice_reference else None,
                            ),
                            self.processor.build_assistant_message(
                                audio_codes_list=previous_audio_codes,
                                content=self.build_audio_placeholder_content(len(previous_audio_codes)),
                            ),
                        ]]
                        processor_mode = "continuation"
                    elif voice_reference:
                        conversations = [[
                            self.processor.build_user_message(
                                text=prompt,
                                language=language or None,
                                reference=[voice_reference],
                            )
                        ]]
                        processor_mode = "generation"
                    else:
                        conversations = [[self.processor.build_user_message(text=prompt, language=language or None)]]
                        processor_mode = "generation"

                    outputs = self.generate_outputs(conversations, processor_mode, temperature, audio_top_p, audio_top_k)
                    decoded = self.decode_outputs_to_sounds_and_audio(outputs)

                    if isinstance(decoded, str):
                        return decoded

                    prompt_sounds, prompt_audios = decoded
                    sounds.extend(prompt_sounds)
                    if prompt_audios:
                        self.cache_continuation(
                            prompt,
                            prompt_audios[-1],
                            prompt_sounds[-1].sr,
                            rolling_continuation_max_segments,
                        )

                return sounds

            if voice_reference:

                conversations = [
                    [
                        self.processor.build_user_message(
                            text=prompt,
                            language=language or None,
                            reference=[voice_reference],
                        )
                    ]
                    for prompt in prompts
                ]
                processor_mode = "generation"

            else:
                
                # No voice clone data
                conversations = [
                    [self.processor.build_user_message(text=prompt, language=language or None)]
                    for prompt in prompts
                ]
                processor_mode = "generation"

            outputs = self.generate_outputs(conversations, processor_mode, temperature, audio_top_p, audio_top_k)
            decoded = self.decode_outputs_to_sounds(outputs)

            if isinstance(decoded, str):
                # Return single error string
                return decoded 

            # Return list of Sounds
            return decoded

        except Exception as e:
            
            return make_error_string(e)

MOSS_AUDIO_PLACEHOLDER = "<|audio|>"
