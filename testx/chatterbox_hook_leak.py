"""
Repro/reference for the Chatterbox Multilingual per-segment memory leak
(GitHub issue #38) and validation of the app-side fix.

Root cause: chatterbox's T3.inference() resets `self.compiled = False` on
every call, so every generate() constructs a fresh AlignmentStreamAnalyzer,
which registers a forward hook on each of three attention layers of the T3
transformer. The hook handles are discarded and the hooks are never removed.
Each generated segment therefore leaves three live hooks plus an orphaned
analyzer (retaining CPU tensors) attached to the model — worker RSS grows per
segment until the model is torn down (Options > Unload models), matching the
issue's report. The stale hooks also copy attention maps to the CPU on every
subsequent decode step of every later generation.

ChatterboxModel.generate() strips these hooks after each call; this script
demonstrates the leaky behavior with the strip disabled and the flat behavior
with it enabled.

Run (needs the Chatterbox model files in the HF cache and a CUDA GPU):

    ./venv-cb/bin/python -m testx.chatterbox_hook_leak
"""

import argparse
import gc
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import psutil
import torch

TEXTS = [
    "The old lighthouse keeper climbed the winding stairs each evening to trim the wick and light the lamp.",
    "Rain drummed softly on the greenhouse roof while the gardener sorted seeds into small paper envelopes.",
    "A single lantern swayed above the dock as the last fishing boat slipped quietly out of the harbor.",
]

PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def make_reference_wav(path: str, source_path: str, seconds: float = 10.0) -> None:
    import wave
    with wave.open(source_path) as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    reps = int(seconds * rate / len(data)) + 1
    long_data = np.tile(data, reps)[: int(seconds * rate)]
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(long_data.tobytes())


def tensor_bytes(t) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


def analyzer_report(model) -> tuple[int, float, dict[int, int]]:
    from chatterbox.models.t3.inference.alignment_stream_analyzer import (
        AlignmentStreamAnalyzer,
    )
    from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel

    analyzers = []
    for o in gc.get_objects():
        try:
            if isinstance(o, AlignmentStreamAnalyzer):
                analyzers.append(o)
        except ReferenceError:
            continue
    total = sum(
        sum(tensor_bytes(t) for t in a.last_aligned_attns if t is not None)
        + tensor_bytes(a.alignment)
        for a in analyzers
    )
    hooks = {
        idx: len(model.t3.tfmr.layers[idx].self_attn._forward_hooks)
        for idx in ChatterboxModel._ALIGNED_ATTN_LAYER_INDICES
    }
    return len(analyzers), total / 1e6, hooks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--apply-fix", action="store_true")
    args = parser.parse_args()

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel

    asset = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tts_audiobook_tool", "assets", "done.wav",
    )
    make_reference_wav("chatterbox_hook_leak_ref.wav", asset)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    model.prepare_conditionals("chatterbox_hook_leak_ref.wav")

    label = "fixed" if args.apply_fix else "leaky"
    print(f"=== {label}: {args.iterations} generations ===", flush=True)
    for i in range(args.iterations):
        t0 = time.time()
        wav = model.generate(TEXTS[i % len(TEXTS)], language_id="en")
        _ = wav.cpu().numpy().squeeze()
        if args.apply_fix:
            # What ChatterboxModel.generate() does after each call:
            ChatterboxModel._strip_alignment_analyzer_hooks(model)
        n, retained, hooks = analyzer_report(model)
        print(
            f"gen {i + 1:2d}: {time.time() - t0:5.2f}s rss={rss_mb():8.1f} MB "
            f"analyzers={n:3d} analyzer_retained={retained:7.3f} MB hooks={hooks}",
            flush=True,
        )

    try:
        os.remove("chatterbox_hook_leak_ref.wav")
    except OSError:
        pass


if __name__ == "__main__":
    main()
