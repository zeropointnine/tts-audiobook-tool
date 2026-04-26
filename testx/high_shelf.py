from pathlib import Path
import soundfile

from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil


"""
SoundUtil.high_shelf_eq() 
Test script
"""

def _fmt(v: float) -> str:
    """Filename-friendly numeric formatting."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "neg_").replace(".", "p")


def main() -> int:

    source_path = Path("/d/w/w/goneworld/sample.flac")

    # Suggested defaults for mildly muffled 24kHz TTS
    strength = 0.8
    boost_start_hz = 4000.0
    q_like = 1.2

    load_result = SoundFileUtil.load(str(source_path))
    if isinstance(load_result, str):
        print(f"Couldn't load {source_path}: {load_result}")
        return 1

    processed_sound = SoundUtil.high_shelf_eq(
        load_result,
        strength=strength,
        boost_start_hz=boost_start_hz,
        q_like=q_like,
    )

    suffix = (
        f"_str_{_fmt(strength)}"
        f"_boost_start_{_fmt(boost_start_hz)}"
        f"_qlike_{_fmt(q_like)}"
    )
    dest_path = source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}")

    soundfile.write(str(dest_path), processed_sound.data, processed_sound.sr)
    print(f"Saved: {dest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())