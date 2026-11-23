"""OmniVoice LoRA adapters can be applied without peft or an omnivoice upgrade.

Upstream OmniVoice gained first-party LoRA support on 2026-08-05 (k2-fsa/OmniVoice
PR #243): training via PEFT, `omnivoice-infer --lora_adapter`,
`omnivoice-merge-lora`, and `omnivoice.utils.lora.load_lora_adapter()`. None of
that is in a PyPI wheel yet (0.2.1 predates it), and `load_lora_adapter()` is
just `PeftModel.from_pretrained(model, path).merge_and_unload()`.

A first-party adapter directory holds `adapter_config.json` +
`adapter_model.safetensors`. The safetensors keys are relative to the whole
`OmniVoice` module:

    base_model.model.llm.layers.N.{self_attn,mlp}.<proj>.lora_{A,B}.weight
    base_model.model.audio_embeddings.weight   (full replacement, modules_to_save)
    base_model.model.audio_heads.weight        (full replacement, modules_to_save)

So a merge is `W += (alpha / r) * B @ A` (alpha / sqrt(r) for rsLoRA) for each
LoRA pair, plus copying the two replacement tensors. The app does that by hand
(`tts_audiobook_tool/tts_models/omnivoice_lora.py`); this script applies it to
the installed omnivoice (0.1.5) and generates one auto-voice line with and
without the adapter.

Run: ./venv-omni/bin/python -m testx.omnivoice_lora_poc [adapter]
Adapter: a local dir, an hf repo id, or `owner/repo/sub/dir`.
Default adapter: My-Weird-Prompts/Herman-LoRA-V3 (single English speaker,
CC-BY-NC). Outputs go to /tmp/omnivoice_lora_poc_*.wav.
"""

import sys

import soundfile as sf
import torch
from omnivoice import OmniVoice  # type: ignore

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models.omnivoice_lora import load_lora

DEFAULT_ADAPTER = "My-Weird-Prompts/Herman-LoRA-V3"
TEXT = "I've read the press release so you don't have to, and honestly, it raises more questions than it answers."
SEED = 1234


def generate(model: OmniVoice, path: str) -> None:
    torch.manual_seed(SEED)
    audio = model.generate(text=TEXT, language="en")[0]
    sample_rate = model.sampling_rate or 24000
    sf.write(path, audio, sample_rate)
    print(f"wrote {path} ({len(audio) / sample_rate:.1f} s)")


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ADAPTER
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    generate(model, "/tmp/omnivoice_lora_poc_base.wav")
    print(f"merged {load_lora(model, ModelTarget.infer(adapter)).describe()}")
    generate(model, "/tmp/omnivoice_lora_poc_lora.wav")


if __name__ == "__main__":
    main()
