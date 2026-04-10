"""
Quick test script to load the Chatterbox model and run generate.
"""
import sys
import torch

from tts_audiobook_tool.tts_model.chatterbox_model import ChatterboxModel
from tts_audiobook_tool.tts_model.chatterbox_base_model import ChatterboxType

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_type = ChatterboxType.MULTILINGUAL
    print(f"Loading model: {model_type.label} ...")

    model = ChatterboxModel(model_type=model_type, device=device)
    print("Model loaded successfully.")

    text = "-"
    voice_path = ""

    print(f"\n--- Generating audio for: {text!r} with voice: {voice_path} ---")
    try:
        result = model.generate(
            text=text,
            voice_path=voice_path,
            exaggeration=0.5,
            cfg=0.5,
            temperature=0.8,
            top_p=0.95,
            turbo_top_k=-1,
            repetition_penalty=2.0,
            seed=42,
            language_id="en",
        )

        if isinstance(result, str):
            print(f"ERROR (returned string): {result}", file=sys.stderr)
        else:
            print(f"Success! Audio shape: {result.data.shape}, sample rate: {result.sr}, duration: {result.duration:.2f}s")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
