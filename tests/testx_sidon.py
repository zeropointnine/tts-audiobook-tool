import os
import soundfile as sf
import torch

from tts_audiobook_tool.sidon_util import SidonUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil

# ----------------------------------------

INPUT_PATH = "/d/w/w/rebuild6a/combined/long test.m4b"

# ----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

sound = SoundFileUtil.load(INPUT_PATH)
assert not isinstance(sound, str), f"Failed to load: {sound}"
print(f"loaded: {sound.sr} Hz, {sound.duration:.2f}s")

print("loading Sidon models...")
util = SidonUtil()

print("processing...")
result = util.process(sound)
assert not isinstance(result, str), f"Sidon failed: {result}"
print(f"output: {result.sr} Hz, {result.duration:.2f}s")

base, ext = os.path.splitext(INPUT_PATH)
output_path = f"{base}_upscaled{ext}"
sf.write(output_path, result.data, result.sr)
print(f"saved: {output_path}")

util.kill()
