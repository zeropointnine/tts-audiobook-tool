"""Compare objective loudness measurements for two rendered audiobook files.

Edit AUDIO_PATH_A and AUDIO_PATH_B below, then run this file directly. The
analysis uses FFmpeg's loudnorm filter, matching the measurement implementation
used by the application's final loudness-normalization pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any


# ---------------------------------------------------------------------------
# EDIT THESE TWO PATHS
# ---------------------------------------------------------------------------
AUDIO_PATH_A = Path("/d/w/w/kage4/combined/260813_121929 disabled/kage4 [chatterbox] [jy_futaba_s01e08_d].abr.flac")
AUDIO_PATH_B = Path("/d/w/w/kage4/combined/260813_122817 stronger b/kage4 [chatterbox] [jy_futaba_s01e08_d].abr.flac")


@dataclass(frozen=True)
class LoudnessMeasurements:
    duration_seconds: float
    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    gate_threshold_lufs: float


class AnalysisError(RuntimeError):
    pass


def run_command(command: list[str], tool_name: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        raise AnalysisError(f"Could not run {tool_name}: {error}") from error


def measure_duration(path: Path) -> float:
    process = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        "ffprobe",
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or "no error details returned"
        raise AnalysisError(f"ffprobe could not read {path}: {detail}")

    try:
        duration = float(process.stdout.strip())
    except ValueError as error:
        raise AnalysisError(
            f"ffprobe returned an invalid duration for {path}: {process.stdout.strip()!r}"
        ) from error

    if not math.isfinite(duration) or duration <= 0:
        raise AnalysisError(f"Invalid duration for {path}: {duration}")
    return duration


def extract_loudnorm_json(stderr: str) -> dict[str, Any]:
    """Find the loudnorm JSON object without relying on it being the final text."""
    decoder = json.JSONDecoder()
    required_keys = {"input_i", "input_lra", "input_tp", "input_thresh"}

    for index in range(len(stderr) - 1, -1, -1):
        if stderr[index] != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stderr[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and required_keys.issubset(candidate):
            return candidate

    raise AnalysisError(
        "FFmpeg completed without a recognizable loudnorm JSON report.\n"
        f"FFmpeg output:\n{stderr.strip()}"
    )


def finite_measurement(stats: dict[str, Any], key: str, path: Path) -> float:
    try:
        value = float(stats[key])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError(
            f"Invalid or missing loudnorm measurement {key!r} for {path}"
        ) from error

    if not math.isfinite(value):
        raise AnalysisError(
            f"Non-finite loudnorm measurement {key!r} for {path}. "
            "The selected audio may be silent or too short to measure."
        )
    return value


def measure_loudness(path: Path) -> LoudnessMeasurements:
    if not path.is_file():
        raise AnalysisError(f"Audio file does not exist: {path}")

    duration = measure_duration(path)
    process = run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-nostats",
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ],
        "ffmpeg",
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or "no error details returned"
        raise AnalysisError(f"FFmpeg could not analyze {path}:\n{detail}")

    stats = extract_loudnorm_json(process.stderr)
    return LoudnessMeasurements(
        duration_seconds=duration,
        integrated_lufs=finite_measurement(stats, "input_i", path),
        loudness_range_lu=finite_measurement(stats, "input_lra", path),
        true_peak_dbtp=finite_measurement(stats, "input_tp", path),
        gate_threshold_lufs=finite_measurement(stats, "input_thresh", path),
    )


def format_duration(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def print_file_report(label: str, path: Path, stats: LoudnessMeasurements) -> None:
    print(f"{label}: {path}")
    print(f"  Duration              {format_duration(stats.duration_seconds)}")
    print(f"  Integrated loudness   {stats.integrated_lufs:8.2f} LUFS")
    print(f"  Loudness range        {stats.loudness_range_lu:8.2f} LU")
    print(f"  True peak             {stats.true_peak_dbtp:8.2f} dBTP")
    print(f"  Gate threshold        {stats.gate_threshold_lufs:8.2f} LUFS")


def print_delta_report(a: LoudnessMeasurements, b: LoudnessMeasurements) -> None:
    print("B minus A:")
    print(f"  Duration              {b.duration_seconds - a.duration_seconds:+8.3f} s")
    print(f"  Integrated loudness   {b.integrated_lufs - a.integrated_lufs:+8.2f} LU")
    print(f"  Loudness range        {b.loudness_range_lu - a.loudness_range_lu:+8.2f} LU")
    print(f"  True peak             {b.true_peak_dbtp - a.true_peak_dbtp:+8.2f} dB")
    print(f"  Gate threshold        {b.gate_threshold_lufs - a.gate_threshold_lufs:+8.2f} LU")


def print_interpretation() -> None:
    print(
        """
Interpretation:
  * Integrated LUFS is the main whole-file perceived-loudness measurement.
    A less-negative value is louder; a +2 LU delta means B is about 2 dB louder.
  * LRA describes variation over time. Lower can mean more consistent dynamics,
    but whole-file LRA does not directly measure paragraph-to-paragraph matching.
  * True peak estimates inter-sample peaks. Less-negative values are closer to
    clipping; the app's ACX-standard preset currently targets no more than -3 dBTP.
  * Gate threshold is an internal loudness-gating diagnostic, not a tuning target.
  * Compare files made from identical content. Also disable any player-side volume
    normalization when doing the listening comparison.
""".strip()
    )


def main() -> int:
    missing_tools = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing_tools:
        print(f"Missing required executable(s) on PATH: {', '.join(missing_tools)}")
        return 1

    try:
        print("Analyzing A...")
        stats_a = measure_loudness(AUDIO_PATH_A)
        print("Analyzing B...")
        stats_b = measure_loudness(AUDIO_PATH_B)
    except AnalysisError as error:
        print(f"Error: {error}")
        return 1

    print()
    print_file_report("A", AUDIO_PATH_A, stats_a)
    print()
    print_file_report("B", AUDIO_PATH_B, stats_b)
    print()
    print_delta_report(stats_a, stats_b)
    print()
    print_interpretation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
