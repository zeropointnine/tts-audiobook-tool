import torch
import outetts
from outetts.models.config import GenerationConfig
from outetts.models.info import GenerationType

"""
Refer to the Oute TTS project page for more on configuration options, etc.
https://github.com/edwko/OuteTTS
"""

MODEL_CONFIG = outetts.ModelConfig.auto_config(
    model=outetts.Models.VERSION_1_0_SIZE_1B,
    backend=outetts.Backend.HF
)

# Note: `text`, `sampler_config`, and `speaker` properties are set dynamically at runtime
GENERATION_CONFIG = GenerationConfig(
    text = "",
    generation_type= GenerationType.REGULAR
)

# ------------------------------------------------------
# Examples of some other configurations for MODEL_CONFIG

_EXAMPLE_AUTO_HF = outetts.ModelConfig.auto_config(
    model=outetts.Models.VERSION_1_0_SIZE_1B,
    backend=outetts.Backend.HF
)

_EXAMPLE_AUTO_LLAMA = outetts.ModelConfig.auto_config(
    model=outetts.Models.VERSION_1_0_SIZE_1B,
    backend=outetts.Backend.LLAMACPP,
    quantization=outetts.LlamaCppQuantization.FP16
)

_EXAMPLE_MANUAL_HF_WITH_FLASH_ATTN = outetts.ModelConfig(
    model_path="OuteAI/Llama-OuteTTS-1.0-1B",
    tokenizer_path="OuteAI/Llama-OuteTTS-1.0-1B",
    interface_version=outetts.InterfaceVersion.V3,
    backend=outetts.Backend.HF,
    additional_model_config={
        "attn_implementation": "flash_attention_2"
    },
    device="cuda",
    dtype=torch.bfloat16
)

_EXAMPLE_MANUAL_EXL2 = outetts.ModelConfig(
    model_path="local/path/to/oute_tts_model",
    interface_version=outetts.InterfaceVersion.V3,
    backend=outetts.Backend.EXL2,
    device="cuda",
    dtype=torch.bfloat16
)
