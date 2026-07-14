import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.util import printt


def get_flac_paths(directory: Path, recursive: bool=False) -> list[Path]:
    pattern = "**/*.flac" if recursive else "*.flac"
    return sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file()
            and "[raw]" not in path.name
            and "[trim]" not in path.name
            and "[pre_gap_limit]" not in path.name
            and "[post_gap_limit]" not in path.name
        ),
        key=lambda path: str(path).lower(),
    )


def get_tail_dbfs(sound: Sound, end_window_ms: int) -> float:
    if sound.data.size == 0 or sound.sr <= 0 or end_window_ms <= 0:
        return -math.inf

    data = sound.data.astype(np.float32, copy=False)
    mono = SoundExtraUtil._to_mono_float(data)
    if mono.size == 0:
        return -math.inf

    end_window_samples = int(sound.sr * end_window_ms / 1000)
    if end_window_samples <= 0:
        end_window_samples = 1

    tail = mono[-min(mono.size, end_window_samples):]
    tail_rms = SoundExtraUtil._rms(tail)
    if tail_rms <= 0:
        return -math.inf

    return float(20.0 * np.log10(tail_rms))


def format_dbfs(dbfs: float) -> str:
    if math.isinf(dbfs) and dbfs < 0:
        return "-inf"
    return f"{dbfs:.1f}"


def format_path(path: Path) -> str:
    path_str = str(path)
    return make_terminal_hyperlink(path_str, path_str, is_file=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check FLAC files in a directory with SoundExtraUtil.has_hot_tail()."
    )
    parser.add_argument("directory", help="Directory containing FLAC files to test.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan for FLAC files recursively.",
    )
    parser.add_argument(
        "--end-window-ms",
        type=int,
        default=10,
        help="Final analysis window in milliseconds. Default: 10.",
    )
    parser.add_argument(
        "--rms-threshold-dbfs",
        type=float,
        default=-20.0,
        help="Absolute RMS threshold in dBFS. Default: -20.0.",
    )
    parser.add_argument(
        "--only-truncated",
        action="store_true",
        help="Only print files detected as truncated.",
    )
    args = parser.parse_args()

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {format_path(directory)}")

    sound_paths = get_flac_paths(directory, args.recursive)
    if not sound_paths:
        printt(f"No FLAC files found in {format_path(directory)}")
        return

    truncated_count = 0
    load_error_count = 0

    for sound_path in sound_paths:
        sound = SoundFileUtil.load(str(sound_path))
        if not isinstance(sound, Sound):
            load_error_count += 1
            printt(f"ERROR\t{format_path(sound_path)}\t{sound}")
            continue

        is_truncated = SoundExtraUtil.is_possible_truncation(
            sound,
            end_window_ms=args.end_window_ms,
            rms_threshold_dbfs=args.rms_threshold_dbfs,
        )
        tail_dbfs = get_tail_dbfs(sound, args.end_window_ms)

        if is_truncated:
            truncated_count += 1

        if args.only_truncated and not is_truncated:
            continue

        status = "TRUNCATED" if is_truncated else "ok"
        printt(
            f"{status}\t"
            f"tail={format_dbfs(tail_dbfs)} dBFS\t"
            f"duration={sound.duration:.3f}s\t"
            f"{format_path(sound_path)}"
        )

    printt()
    printt(
        f"Checked {len(sound_paths)} FLAC file(s); "
        f"truncated={truncated_count}; "
        f"load_errors={load_error_count}; "
        f"window={args.end_window_ms}ms; "
        f"threshold={args.rms_threshold_dbfs:.1f} dBFS"
    )


if __name__ == "__main__":
    main()
