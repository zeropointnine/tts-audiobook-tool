import numpy as np
import torch

from qwen_tts import Qwen3TTSModel # type: ignore
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem # type: ignore

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model import Qwen3ModelProtocol, Qwen3Protocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *


class Qwen3Model(Qwen3ModelProtocol):
    """
    App wrapper for `Qwen3TTSModel`
    """

    def __init__(self, model_target: str, device: str): 
        
        super().__init__(info=TtsModelInfos.QWEN3TTS.value)

        self._model_target = model_target
        self._voice_info: tuple[str, str] | None = None
        self._voice_clone_prompt: VoiceClonePromptItem | None = None
        self._device = device

        if device == "cuda":             
            device_map = "cuda:0"
        else:
            device_map = device

        attn_implementation = None
        if device == "cuda":
            try:
                from flash_attn import flash_attn_func # type: ignore
                attn_implementation = "flash_attention_2"
                
                # FYI, can ignore:
                # You are attempting to use Flash Attention 2 without specifying a torch dtype. This might lead to unexpected behaviour
               
            except ImportError:
                # eat silently
                ... 

        self._model: Qwen3TTSModel = Qwen3TTSModel.from_pretrained(
                self._model_target,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

    def kill(self) -> None:
        self._model = None # type: ignore
        self._voice_clone_prompt = None

    @property
    def model_target(self) -> str:
        return self._model_target

    @property
    def model_type(self) -> str:
        return self._model.model.tts_model_type # type: ignore

    @property
    def supported_languages(self) -> list[str]:
        return self._model.get_supported_languages() or []

    @property
    def supported_speakers(self) -> list[str]:
        return self._model.get_supported_speakers() or []
    
    @property
    def generate_defaults(self) -> dict[str, Any]:
        return self._model.generate_defaults

    def clear_voice(self) -> None:
        self._voice_info = None
        self._voice_clone_prompt = None

    def generate_base(
          self, 
          prompts: list[str], 
          voice_info: tuple[str, str], 
          language_code: str,
          temperature: float = -1,
          seed: int = -1
    ) -> list[Sound] | str:
        """
        Generate function for model_type="base" (voice clone-based)
        """
        
        if not prompts or not voice_info[0] or not voice_info[1]:
            return "Missing required parameter"

        if not self._voice_clone_prompt or self._voice_info != voice_info:
            # Cache _voice_clone_prompt 
            try:
                self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                    ref_audio=voice_info[0],
                    ref_text=voice_info[1],
                    x_vector_only_mode=False,
                )[0]
            except Exception as e:
                return f"Couldn't create voice clone for {voice_info[0]} - {make_error_string(e)}"
            self._voice_info = voice_info

        voice_clone_prompts = [self._voice_clone_prompt for _ in prompts]
        
        language = self.resolve_language_code_and_warning(language_code)[0]
        languages = [language for _ in prompts]

        if seed == -1:
            seed = random.randrange(0, 2**32 - 1)
        self._set_seed(seed)

        # Inference code does not print its own feedback, so
        printt("Generating...", dont_reset=True) 

        try:
            wavs, sr = self._model.generate_voice_clone(
                text=prompts,
                voice_clone_prompt=voice_clone_prompts,
                language=languages, 
                temperature=self._resolve_temperature(temperature),
                seed=seed,
                non_streaming_mode=True,
            ) 
            
            # FYI, this output can be ignored:
            # setting `pad_token_id` to `eos_token_id`:2150 for open-end generation.
        
        except Exception as e:
            return str(e)
        
        sounds = [Sound(wav, sr) for wav in wavs]
        return sounds

    def generate_custom_voice(
        self, 
        prompts: list[str], 
        speaker_id: str, 
        instruct: str, 
        language_code: str,
        temperature: float = -1,
        seed: int = -1
    ) -> list[Sound] | str:

        speaker_ids = [speaker_id for _ in prompts]
        language = self.resolve_language_code_and_warning(language_code)[0]
        languages = [language for _ in prompts]
        instructs = [instruct for _ in prompts] if instruct else None

        if seed == -1:
            seed = random.randrange(0, 2**32 - 1)
        self._set_seed(seed)

        # Inference code does not print its own feedback, so
        printt("Generating...", dont_reset=True) 

        try:
            wavs, sr = self._model.generate_custom_voice(
                text=prompts,
                speaker=speaker_ids,
                language=languages,
                instruct = instructs,
                temperature=self._resolve_temperature(temperature),
                seed=seed,
                non_streaming_mode=True,
            ) 
        except Exception as e:
            return str(e)

        sounds = [Sound(wav, sr) for wav in wavs]
        return sounds

    def generate_voice_design(
        self, 
        prompts: list[str], 
        instruct: str, 
        language_code: str,
        temperature: float = -1,
        seed: int = -1
    ) -> list[Sound] | str:

        language = self.resolve_language_code_and_warning(language_code)[0]
        languages = [language for _ in prompts]
        instructs = [instruct for _ in prompts] 

        if seed == -1:
            seed = random.randrange(0, 2**32 - 1)
        self._set_seed(seed)

        # Inference code does not print its own feedback, so
        printt("Generating...", dont_reset=True) 

        try:
            wavs, sr = self._model.generate_voice_design(
                text=prompts,
                language=languages,
                instruct = instructs,
                temperature=self._resolve_temperature(temperature),
                seed=seed,
                non_streaming_mode=True,
            ) 
        except Exception as e:
            return str(e)

        sounds = [Sound(wav, sr) for wav in wavs]
        return sounds

    def _resolve_temperature(self, temperature: float) -> float:
        if temperature == -1:
            resolved_temperature = self._model.generate_defaults.get("temperature", None)
            if resolved_temperature is None:
                resolved_temperature = Qwen3Protocol.TEMPERATURE_FALLBACK_DEFAULT
        else:
            resolved_temperature = temperature
        return resolved_temperature

    def _set_seed(self, seed: int):
        """Sets the random seed for reproducibility across torch, numpy, and random."""
        torch.manual_seed(seed)
        if self._device.startswith("cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

