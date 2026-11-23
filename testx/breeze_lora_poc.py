"""
Breeze TTS 2 per-voice LoRAs (Instavar format) on the real model: loads `BreezeModel` once, then
switches between adapters in place (`tts_audiobook_tool/tts_models/lora_switch.py`), timing each
switch and writing a line per adapter for listening. Then checks exactness on the real weights:
- after A -> B, the targeted weights equal a fresh merge of B onto the checkpoint's base weights
- after switching to no LoRA, they equal the checkpoint's base weights

Accepts either directory form: an adapter export (adapter_config.json + weights) or a training
checkpoint (`.../lora/checkpoint-step-N`, with the run's run-config.json beside it).

Run: ./venv-breeze/bin/python -m testx.breeze_lora_poc ADAPTER_A [ADAPTER_B] [--strength 1.0]
     [--ref clip.wav --ref-text "..."] [--no-pin]
Outputs go to /tmp/breeze_lora_poc_*.wav.
"""
import argparse
import time

import soundfile as sf
import torch

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel, BreezeSampling
from tts_audiobook_tool.tts_models.breeze_model import BreezeModel
from tts_audiobook_tool.tts_models.lora_merge import apply_merge

TEXT = "The lantern swung once in the wind, and then the whole street went quiet."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", nargs="+")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--ref", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-pin", action="store_true", help="Read base weights from the model files on each switch")
    args = parser.parse_args()

    model = BreezeModel(DeviceType.CUDA)
    switcher = model._loras
    assert switcher is not None and model._model is not None
    if args.no_pin:
        switcher.base._pin_policy = lambda *_: False

    loras = [VoiceLora(ModelTarget.infer(target), args.strength) for target in args.adapters]
    for lora in loras:
        error = switcher.prepare(lora)
        if error:
            raise SystemExit(f"LoRA not usable: {lora.target}: {error}")

    local = BreezeBaseModel.LOCAL
    sampling = BreezeSampling(
        temperature=float(local.number("temperature").default), top_k=int(local.number("top_k").default),
        top_p=float(local.number("top_p").default),
        repetition_penalty=float(local.number("repetition_penalty").default), cfg=1.0,
    )
    # Every adapter, then back to the first (a restore + re-merge), then none
    sequence = [*loras, loras[0], None] if len(loras) > 1 else [loras[0], None, loras[0]]
    for step, lora in enumerate(sequence):
        start = time.perf_counter()
        error = switcher.activate(lora)
        if error:
            raise SystemExit(error)
        print(f"switch {step} -> {lora.target if lora else 'none'}: {time.perf_counter() - start:.2f}s "
              f"(base cache: {switcher.base.mode}, {switcher.base.pinned_bytes / 1024 ** 3:.1f} GB pinned)")
        result = model.generate([TEXT], voice_path=args.ref, transcript=args.ref_text if args.ref else "",
                                instructions="", sampling=sampling, seed=args.seed)
        if isinstance(result, str):
            raise SystemExit(result)
        path = f"/tmp/breeze_lora_poc_{step}.wav"
        sf.write(path, result[0].data, result[0].sr)
        print(f"  wrote {path} ({len(result[0].data) / result[0].sr:.1f}s)")

    check_exactness(model, loras)


def check_exactness(model: BreezeModel, loras: list[VoiceLora]) -> None:
    switcher = model._loras
    assert switcher is not None and model._model is not None
    params = dict(model._model.named_parameters())
    last = loras[-1]
    switcher.activate(loras[0])
    switcher.activate(last)
    switcher.prepare(last)
    prepared = switcher._prepared[(last.target, last.strength)]
    names = sorted(prepared.plan.target_names)
    base = dict(switcher.base._source.read(params, names))

    # A fresh merge of `last` onto the checkpoint's base values, one weight at a time
    mismatched = 0
    for name in names:
        scratch = torch.nn.Module()
        scratch.weight = torch.nn.Parameter(base[name].to(params[name].device), requires_grad=False)
        one = type(prepared.plan)(rank=prepared.plan.rank, alpha=prepared.plan.alpha, scale=prepared.plan.scale)
        if name in prepared.plan.pairs:
            one.pairs["weight"] = prepared.plan.pairs[name]
        else:
            one.replacements["weight"] = prepared.plan.replacements[name]
        apply_merge(scratch, one, delta_device="target")
        mismatched += not torch.equal(scratch.weight, params[name])
    print(f"after switches, {last.target} vs fresh merge: {len(names) - mismatched}/{len(names)} weights identical")

    switcher.activate(None)
    restored = sum(torch.equal(base[name].to(params[name].device), params[name]) for name in names)
    print(f"after switching to none: {restored}/{len(names)} weights equal the checkpoint")
    switcher.end_run()
    if mismatched or restored != len(names):
        raise SystemExit("exactness check FAILED")


if __name__ == "__main__":
    main()
