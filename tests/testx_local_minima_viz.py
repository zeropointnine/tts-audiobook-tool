from pathlib import Path
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound_debug_util import SoundDebugUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil

# Test "find_local_minima()"

file_path = "/home/lee/Documents/w/w/succession/segments-test/[00692] [01767548142799] [raw].flac"
target_timestamp = 3.18 + 0.2

sound_or_error = SoundFileUtil.load(file_path)
if not isinstance(sound_or_error, Sound):
    print("error:", sound_or_error)
    exit(0)
sound = sound_or_error

result = SoundUtil.find_local_minima(sound, target_timestamp)
if isinstance(result, str):
    print("error:", result)
    exit(0)
local_minima = result

print(f"Target time: {target_timestamp}")
print(f"Local minima: {local_minima}")

png_path = Path(file_path).with_suffix(".png")
SoundDebugUtil.save_local_minima_visualization(sound, target_timestamp, local_minima, str(png_path))

