import os

from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


SOUND_DIR = "/d/w/moss-music-test"


detector = ModelManager.get_yamnet_detector()

file_names = sorted(os.listdir(SOUND_DIR))
count = 0

for file_name in file_names:

    if not file_name.lower().endswith(".flac"):
        continue

    sound_path = os.path.join(SOUND_DIR, file_name)
    sound = SoundFileUtil.load(sound_path)
    if isinstance(sound, str):
        print(f"Error loading {sound_path}: {sound}")
        continue

    if detector.has_music(sound):
        print(sound_path)

    count += 1

print()
print("Num files examined:", count)