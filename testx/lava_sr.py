"""
Manual LavaSR v2 inference smoke test.

Loads an audio file into the app's Sound representation, restores it with
LavaSrUtil, and saves a 48 kHz result beside the source file. The utility
automatically selects CUDA, MPS, or CPU; MPS failures fall back to CPU.

Usage:
    venv-base/bin/python testx/lava_sr.py INPUT_AUDIO
    venv-base/bin/python testx/lava_sr.py INPUT_AUDIO --denoise

The default performs bandwidth enhancement only. Pass --denoise to run the
LavaSR denoiser before enhancement. Model files may be downloaded from
Hugging Face during the first run. The output filename ends in ``_lava_sr``.
"""

import os
import sys

import soundfile as sf

from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


if len(sys.argv) < 2:
    raise SystemExit(f"Usage: {sys.argv[0]} INPUT_AUDIO [--denoise]")

input_path = sys.argv[1]
denoise = "--denoise" in sys.argv[2:]

sound = SoundFileUtil.load(input_path)
assert not isinstance(sound, str), f"Failed to load: {sound}"
print(f"loaded: {sound.sr} Hz, {sound.duration:.2f}s")

print("loading LavaSR v2 model...")
lava_sr_util = LavaSrUtil()
print(f"device: {lava_sr_util.device}")
print(f"denoise: {denoise}")

try:
    print("processing...")
    result = lava_sr_util.process(sound, denoise=denoise)
    assert not isinstance(result, str), f"LavaSR failed: {result}"
    print(f"output: {result.sr} Hz, {result.duration:.2f}s")

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_lava_sr{ext}"
    sf.write(output_path, result.data, result.sr)
    print(f"saved: {output_path}")
finally:
    lava_sr_util.kill()
