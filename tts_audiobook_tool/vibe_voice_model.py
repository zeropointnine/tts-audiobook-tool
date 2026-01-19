import traceback
import numpy as np
import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference # type: ignore
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor # type: ignore
from peft import PeftModel
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_model import VibeVoiceModelProtocol, VibeVoiceProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *


class VibeVoiceModel(VibeVoiceModelProtocol):
    """
    VibeVoice TTS inference logic
    Mostly copy-pasted from: VibeVoice/demo/inference_from_file.py
    """

    def __init__(
            self,
            device_map: str,
            model_path: str = "",
            lora_path: str | None = None,
            max_new_tokens: int | None = None,
    ):
        super().__init__(TtsModelInfos.VIBEVOICE.value)

        self._device_map = device_map
        if not model_path:
            model_path = VibeVoiceProtocol.DEFAULT_MODEL_PATH
        self.max_new_tokens = max_new_tokens

        self.processor = VibeVoiceProcessor.from_pretrained(model_path)

        # Determine attention implementation type
        if "cuda" in device_map:
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
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_implementation
        )
        
        if lora_path:
            # Load LoRA adapter
            printt()
            printt(f"Loading LoRA adapter from: {lora_path}")
            printt()
            try:
                # The LoRA is trained on the language model backbone (Qwen2)
                # VibeVoiceForConditionalGenerationInference wraps VibeVoiceModel
                # which contains the language_model as a submodule
                # We need to load the adapter onto the language_model submodule
                language_model = self.model.model.language_model
                
                if hasattr(language_model, 'load_adapter'):
                    language_model.load_adapter(lora_path)
                else:
                    # Fallback to PeftModel wrapper
                    printt("load_adapter() not available, using PeftModel wrapper on language_model")
                    self.model.model.language_model = PeftModel.from_pretrained(language_model, lora_path)
                printt("LoRA adapter activated successfully")
                printt()
                self._has_lora = True
            except Exception as e:
                printt(f"{COL_ERROR}WARNING: Failed to load LoRA adapter: {e}")
                printt()
                self._has_lora = False
        else:
            self._has_lora = False
        
        self.model.eval()

    @property
    def has_lora(self) -> bool:
        return self._has_lora

    def kill(self) -> None:
        if self.processor:
            self.processor.tokenizer = None
            self.processor.audio_processor = None
            self.processor.audio_normalizer = None
        self.processor = None
        self.model = None

    def massage_for_inference(self, text: str) -> str:
        text = super().massage_for_inference(text)
        # Required speaker tag
        text = f"{SPEAKER_TAG}{text}" 
        return text

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            cfg_scale: float=VibeVoiceProtocol.CFG_DEFAULT,
            num_steps: int=VibeVoiceProtocol.DEFAULT_NUM_STEPS,
            seed: int = -1
    ) -> list[Sound] | str:
        """
        Returns list[Sound] or error string

        FYI: Couldn't pass pre-loaded sound data without inference issues for some reason,
        but in practice overhead is negligible
        """
        if self.model is None or self.processor is None:
            return "model or processor is not initialized" # logic error

        if seed <= -1:
            seed = random.randrange(0, 2**32 - 1)
        self.set_seed(seed)

        try:
            self.model.set_ddpm_inference_steps(num_steps) # type: ignore

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

            # Generate audio
            outputs = self.model.generate(
                **inputs, # type: ignore
                max_new_tokens=self.max_new_tokens,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
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
            sound = Sound(ndarray_data, TtsModelInfos.VIBEVOICE.value.sample_rate)
            sounds.append(sound)

        return sounds

    def set_seed(self, seed: int):
        """ 
        Sets various static random seed values 
        From Chatterbox inference code 
        """
        torch.manual_seed(seed)
        if self._device_map.startswith("cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


SPEAKER_TAG = "Speaker 1: "
