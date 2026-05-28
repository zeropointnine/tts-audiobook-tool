import os
from pathlib import Path

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.util import printt


DIR = "/mnt/x/w/rebuild6a-alt/segments"
SOUND_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


def make_trimmed_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_TRIMMED.flac")


def main() -> None:
    sound_paths = sorted(
        (
            Path(DIR) / file_name
            for file_name in os.listdir(DIR)
            if (Path(DIR) / file_name).is_file()
            and (Path(DIR) / file_name).suffix.lower() in SOUND_EXTENSIONS
            and not (Path(DIR) / file_name).stem.endswith("_TRIMMED")
        ),
        key=lambda path: path.name.lower(),
    )

    for sound_path in sound_paths:
        sound = SoundFileUtil.load(str(sound_path))
        if not isinstance(sound, Sound):
            printt(f"Skipping {sound_path}: {sound}")
            continue

        trimmed = SoundExtraUtil.trim_trailing_token_noise(sound)
        if len(trimmed.data) == len(sound.data):
            continue

        trimmed_path = make_trimmed_path(sound_path)
        error = SoundFileUtil.save_flac(trimmed, str(trimmed_path))
        if error:
            printt(f"Failed to save {trimmed_path}: {error}")
            continue

        trimmed_ms = (sound.duration - trimmed.duration) * 1000
        printt(f"Trimmed {trimmed_ms:.1f}ms: {sound_path} -> {trimmed_path}", "\n\n")


if __name__ == "__main__":
    main()
