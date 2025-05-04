import outetts
import torch

"""
Refer to the Oute TTS project page for more on configuration options, etc.
https://github.com/edwko/OuteTTS
"""

MODEL_CONFIG = outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.HF,
        quantization=outetts.LlamaCppQuantization.FP16
    )





# ----------------------------------------------------------
# Example of setting all the model config values explicitly, 
# which in this case includes use flash attention.

EXAMPLE_MANUAL_MODEL_CONFIG = outetts.ModelConfig(
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
