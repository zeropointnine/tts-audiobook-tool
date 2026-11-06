"""OmniVoice voice clone prompt creation records an autograd graph.

`OmniVoice.create_voice_clone_prompt()` is not decorated with
`torch.inference_mode()` (unlike `generate()` and `transcribe()`). It encodes
the reference audio with the Higgs Audio v2 tokenizer, whose
`HiggsAudioV2TokenizerModel.encode()` wraps only the semantic model in
`torch.no_grad()`. The acoustic encoder (a DAC conv stack running over the
raw 24 kHz waveform) and the semantic encoder run with autograd enabled, and
the weights load with `requires_grad=True`. So every intermediate activation
is saved for a backward pass that never happens.

The graph is not leaked: the output codes are integer tensors with no
`grad_fn`, so it is freed when `encode()` returns. The problem is the peak.
PyTorch's caching allocator then keeps that peak reserved for the rest of the
session, where CTranslate2 (faster-whisper) and other runtimes sharing the GPU
cannot use it.

Measured on an RTX 3080 Ti, fp32 tokenizer (as OmniVoice loads it), extra
peak allocation over the loaded weights:

     1 s reference    as called:  +359 MiB    inference_mode:  +58 MiB
     2 s reference    as called:  +659 MiB    inference_mode:  +60 MiB
    16 s reference    as called:  (OOM with ~2 GiB free)    inference_mode: +473 MiB

That is ~300 MiB per second of reference audio as called, extrapolating to
~4.5 GiB for a 16 s reference (PR #48 measured 6807 vs 2419 MiB peak for the
whole fp16 model), versus ~30 MiB per second under `inference_mode()`. The
codes are bit-identical either way. `OmniVoiceModel._create_voice_clone()` now wraps the
call in `torch.inference_mode()` (PR #48, issue #27).

Loads only the audio tokenizer from the k2-fsa/OmniVoice repo, which is the
part of `create_voice_clone_prompt()` that matters here. Long durations need a
lot of free VRAM in the as-called mode; a duration that runs out of memory is
reported and skipped.

Run: ./venv-omni/bin/python -m testx.omnivoice_clone_prompt_vram
"""

import contextlib
import os

import torch
from huggingface_hub import snapshot_download
from transformers import HiggsAudioV2TokenizerModel

REPO = "k2-fsa/OmniVoice"
DURATIONS_SECONDS = (1, 2, 4, 8, 16)
MIB = 2**20


def load_tokenizer() -> HiggsAudioV2TokenizerModel:
    root = snapshot_download(REPO, allow_patterns=["audio_tokenizer/*"])
    return HiggsAudioV2TokenizerModel.from_pretrained(
        os.path.join(root, "audio_tokenizer"), device_map="cuda"
    )


def measure(tokenizer, audio: torch.Tensor, ctx) -> tuple[float, torch.Tensor]:
    """Returns (extra peak MiB over the baseline allocation, codes on CPU)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with ctx():
        codes = tokenizer.encode(audio).audio_codes
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MIB
    return peak, codes.cpu().clone()


def main() -> None:
    tokenizer = load_tokenizer()
    param = next(tokenizer.parameters())
    sr = tokenizer.config.sample_rate
    hop = tokenizer.config.hop_length
    print(f"dtype={param.dtype}  requires_grad={param.requires_grad}  sr={sr}\n")

    torch.manual_seed(0)
    for seconds in DURATIONS_SECONDS:
        n = (seconds * sr) // hop * hop
        audio = (torch.randn(1, 1, n) * 0.1).cuda()
        # inference_mode first so its number is not affected by a prior OOM
        fixed_peak, fixed_codes = measure(tokenizer, audio, torch.inference_mode)
        try:
            as_called_peak, as_called_codes = measure(tokenizer, audio, contextlib.nullcontext)
            same = torch.equal(fixed_codes, as_called_codes)
            print(f"{seconds:3d} s  as called: +{as_called_peak:6.0f} MiB"
                  f"   inference_mode: +{fixed_peak:5.0f} MiB   codes identical: {same}",
                  flush=True)
        except torch.OutOfMemoryError:
            print(f"{seconds:3d} s  as called: out of memory"
                  f"   inference_mode: +{fixed_peak:5.0f} MiB", flush=True)
        del audio


if __name__ == "__main__":
    main()
