"""
Breeze TTS 2 LoRA (Instavar format) on the real model: loads `BreezeModel` with an
adapter merged in (`tts_audiobook_tool/tts_models/breeze_lora.py`), reports the merge,
and writes a sample-less line and (optionally) a voice-clone line for listening.

Accepts either directory form: an adapter export (adapter_config.json + weights) or a
training checkpoint (`.../lora/checkpoint-step-N`, with the run's run-config.json beside it).

Run: ./venv-breeze/bin/python -m testx.breeze_lora_poc ADAPTER [--strength 1.0] [--ref clip.wav --ref-text "..."]
Outputs go to /tmp/breeze_lora_poc_*.wav.
"""
import argparse

import soundfile as sf

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel, BreezeSampling
from tts_audiobook_tool.tts_models.breeze_model import BreezeModel

TEXT = "The lantern swung once in the wind, and then the whole street went quiet."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--ref", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    model = BreezeModel(DeviceType.CUDA, lora_target=args.adapter, lora_strength=args.strength)
    if not model.has_lora:
        raise SystemExit(f"LoRA not loaded: {model.lora_error}")

    local = BreezeBaseModel.LOCAL
    sampling = BreezeSampling(
        temperature=float(local.number("temperature").default), top_k=int(local.number("top_k").default),
        top_p=float(local.number("top_p").default),
        repetition_penalty=float(local.number("repetition_penalty").default), cfg=1.0,
    )
    cases = [("plain", "", "")]
    if args.ref:
        cases.append(("ref", args.ref, args.ref_text))
    for name, voice_path, transcript in cases:
        result = model.generate([TEXT], voice_path=voice_path, transcript=transcript, instructions="",
                                sampling=sampling, seed=args.seed)
        if isinstance(result, str):
            raise SystemExit(result)
        path = f"/tmp/breeze_lora_poc_{name}.wav"
        sf.write(path, result[0].data, result[0].sr)
        print(f"wrote {path} ({len(result[0].data) / result[0].sr:.1f}s)")


if __name__ == "__main__":
    main()
