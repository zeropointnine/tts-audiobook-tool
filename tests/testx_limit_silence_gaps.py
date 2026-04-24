"""
Test script for SilenceUtil.enforce_max_silence.
Iterates audio files in a hard-coded directory, applies enforce_max_silence,
and reports/saves results.
"""

import os
import sys
import soundfile

# ── Adjust this path as needed ───────────────────────────────────
AUDIO_DIR = "/d/w/w/downbelow/temp delete"
# ─────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.silence_util import SilenceUtil


def enforce_max_silence_for_file(file_path: str, max_silence_seconds: float):
    data, sr = soundfile.read(file_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mix down to mono
    sound = Sound(data, int(sr))

    original_duration = sound.duration
    trimmed = SilenceUtil.limit_silence_gaps(sound, max_silence_seconds)
    new_duration = trimmed.duration

    basename = os.path.basename(file_path)
    root, ext = os.path.splitext(basename)

    if abs(new_duration - original_duration) < 1e-3:
        return

    print(basename)
    print(f"  {original_duration:.3f}s → {new_duration:.3f}s  (Δ {original_duration - new_duration:.3f}s)")

    out_name = f"{root}_TRIMMED{ext}"
    out_path = os.path.join(os.path.dirname(file_path), out_name)
    soundfile.write(out_path, trimmed.data, trimmed.sr)
    print(f"  Saved → {out_path}")


def main():
    max_silence_seconds = 1.0

    if not os.path.isdir(AUDIO_DIR):
        print(f"Directory not found: {AUDIO_DIR}")
        sys.exit(1)

    audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".m4b", ".ogg", ".aac"}
    files = sorted(
        f for f in os.listdir(AUDIO_DIR)
        if os.path.splitext(f)[1].lower() in audio_extensions
    )

    if not files:
        print(f"No audio files found in {AUDIO_DIR}")
        sys.exit(0)

    for filename in files:
        file_path = os.path.join(AUDIO_DIR, filename)
        try:
            enforce_max_silence_for_file(file_path, max_silence_seconds)
        except Exception as e:
            print(f"{filename}")
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    main()