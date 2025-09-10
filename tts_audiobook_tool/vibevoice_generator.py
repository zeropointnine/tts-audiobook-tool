import torch
from transformers.utils import logging
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.tts_info import VIBEVOICE_SPECS
from tts_audiobook_tool.util import printt

#z
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class VibeVoiceGenerator:
    """
    Logic comes from: vibe voice project command line script file "inference_from_file.py"
    """

    def __init__(self, device_map: str, max_new_tokens: int | None = None):

        self.max_new_tokens = 1000 #z max_new_tokens

        # Load processor
        model_path = "microsoft/VibeVoice-1.5b"
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)

        # Determine attention implementation type
        attn_implementation = "sdpa" # fyi, in my testing, "eager" is completely unusable
        if "cuda" in device_map:
            try:
                from flash_attn import flash_attn_func
                attn_implementation = "flash_attention_2"
            except ImportError as e:
                printt("\nWarning: Flash attention is not installed. Will fall back to sdpa.\n")
        printt()
        printt(f"Attention implementation: {attn_implementation}")
        printt()

        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_implementation
        )
        self.model.eval()

    def kill(self) -> None:
        self.processor = None
        self.model = None

    def preprocess_text(self, text: str) -> str:

        # Fancy apost causes rest of word to be omitted (eg, "She's" ->  "She")
        text = text.replace("’", "'")

        # Em dash and semicolon can result in no 'caesura'
        text = text.replace("—", ", ")
        text = text.replace("─", ", ") # Saw this used in the wild in a commercial epub
        text = text.replace(";", ",")

        # Ellipsis character can wreck gen badly. Triple-dot as well.
        text = text.replace("…", ".")
        text = text.replace("...", ".")

        # FYI, colon pretty reliably produces caesura but it depends

        text = TextUtil.un_all_caps(text)

        # Required speaker tag
        text = f"{SPEAKER_TAG}{text}"

        return text

    def generate(self, voice_path: str, text: str, cfg_scale: float=DEFAULT_CFG_VIBEVOICE, num_steps: int=DEFAULT_NUM_STEPS_VIBE_VOICE) -> Sound | str:
        """
        Returns Sound or error string

        FYI: Couldn't pass pre-loaded sound data without inference issues for some reason,
        but in practice overhead is negligible
        """

        assert(self.model is not None)
        assert(self.processor is not None)

        self.model.set_ddpm_inference_steps(num_steps)

        text = self.preprocess_text(text)

        #z
        printt(f"\n[{text}]\n")

        inputs = self.processor(
            text=[ text ],  # Wrap in list for batch processing
            voice_samples=[ [ voice_path ] ],  # Wrap in list for batch processing
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

        has_audio = outputs.speech_outputs and outputs.speech_outputs[0] is not None # type: ignore
        if not has_audio:
            return "No audio output"

        # First (and only) batch item. A tuple of 3 items.
        data = outputs.speech_outputs[0] # type: ignore

        # Is bfloat16
        tensor_data = data[0]
        if tensor_data.dtype == torch.bfloat16:
            tensor_data = tensor_data.to(torch.float32)

        ndarray_data = tensor_data.cpu().numpy()
        sound = Sound(ndarray_data, VIBEVOICE_SPECS.sample_rate)
        return sound


SPEAKER_TAG = "Speaker 1: "
