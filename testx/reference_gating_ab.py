"""
Hard-gated voice-reference harness (dev tool).

Background: a voice-clone reference whose interior pauses are digitally silent
(no room-tone floor) can make an in-context TTS model never emit its
end-of-audio token. On audio.cpp's Higgs TTS that surfaces as HTTP 500
"reached max_tokens ... before EOC", and the app then retries the same line,
so one pathological reference produces a storm of failures. Measured on
2026-10-07: 5/20 requests failed with such a reference, 0/20 with the same
speech re-floored at -75 dBFS (pairs fixed, same seeds and texts).

Subcommands
-----------
  detect  Scan files/dirs for hard-gated silence. No server needed.
  floor   Write a copy with the gated stretches filled by room-tone-level
          noise (tts_audiobook_tool.sound.hard_gate_util.HardGateUtil).
          Pause count/lengths/positions are preserved: only the sample values
          inside the filled runs change.
  ab      Run the same texts and seeds against two references and report
          failure rate, output noise floor and (optionally) WER for each arm.

Examples
--------
  ./venv-base/bin/python -m testx.reference_gating_ab detect /d/w/voice
  ./venv-base/bin/python -m testx.reference_gating_ab floor in.flac --floor-db -65
  ./venv-base/bin/python -m testx.reference_gating_ab ab \
      --ref-a in.flac --ref-b in_floored.flac \
      --server http://127.0.0.1:8080 --model higgs-audio-tts \
      --reference-text-file ref.txt --texts texts.txt --seeds 3

The 'ab' payload mirrors what the app sends to an audio.cpp server
(/v1/audio/speech: base64 mono PCM16 WAV voice_ref, response_format=wav,
language, sampler settings, request options). Defaults for --temperature,
--top-p and --top-k and the pinned text_chunk_size option mirror the
higgs_v3_audiocpp catalog entry; override them for other families. The exact
payload used is printed and saved to results.json so a run is reproducible.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile

if TYPE_CHECKING:
    import httpx

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.hard_gate_util import (
    HARD_GATING_MIN_GAP_MS,
    HARD_GATING_NEAR_ZERO_AMPLITUDE,
    HardGatingInfo,
    HardGateUtil,
)
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
DEFAULT_TEXTS = [
    "The sun had not yet risen. The sea was indistinguishable from the sky.",
    "It is this cruelty that has, in the past, made it easy for me to see.",
    "There is a blackish spot on the fleshy base of my thumb.",
]


# --------------------------------------------------------------------------
# loading / detection
# --------------------------------------------------------------------------

def load_sound(path: str) -> Sound | str:
    """ App-preferred loader, with an ffmpeg fallback for containers librosa
    cannot open (e.g. .m4a). """
    loaded = SoundFileUtil.load(path)
    if isinstance(loaded, Sound):
        return loaded
    decoded = _decode_with_ffmpeg(path)
    return decoded if decoded is not None else loaded


def _decode_with_ffmpeg(path: str) -> Sound | None:
    if shutil.which("ffmpeg") is None:
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    if not result.stdout:
        return None
    return Sound(np.frombuffer(result.stdout, dtype=np.float32), 48000)


def inspect_gating(sound: Sound, args: argparse.Namespace) -> HardGatingInfo:
    return HardGateUtil.inspect(
        sound,
        near_zero_amplitude=args.near_zero,
        min_gap_ms=args.min_gap_ms,
    )


def collect_audio_paths(paths: list[str]) -> list[str]:
    found: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for dirpath, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if Path(name).suffix.lower() in AUDIO_EXTS:
                        found.append(os.path.join(dirpath, name))
        elif path.is_file():
            found.append(str(path))
        else:
            print(f"skip (not found): {raw}", file=sys.stderr)
    return found


def print_info(label: str, info: HardGatingInfo) -> None:
    severity = "SEVERE" if info.is_severe else ("gated" if info.is_gated else "clean")
    print(f"  {label}: [{severity}] {info.describe()} | "
          f"duration {info.duration:.2f}s, exactly-zero samples {info.absolute_zero_pct:.1f}%")


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace) -> int:
    paths = collect_audio_paths(args.paths)
    if not paths:
        print("No audio files found.")
        return 1
    rows: list[tuple[str, HardGatingInfo | str]] = []
    for path in paths:
        sound = load_sound(path)
        rows.append((path, inspect_gating(sound, args) if isinstance(sound, Sound) else sound))
    rows.sort(key=lambda row: -(row[1].longest_gap_ms if isinstance(row[1], HardGatingInfo) else -1.0))

    print(f"{'file':70s} {'dur':>6s} {'gated%':>7s} {'zero%':>7s} {'longest':>8s} {'floor':>8s}  verdict")
    for path, info in rows:
        if isinstance(info, str):
            print(f"{path[:70]:70s} {'-':>6s} {'-':>7s} {'-':>7s} {'-':>8s} {'-':>8s}  undecodable: {info[:40]}")
            continue
        verdict = "SEVERE" if info.is_severe else ("gated" if info.is_gated else "clean")
        print(f"{path[:70]:70s} {info.duration:6.2f} {info.gated_sample_pct:7.1f} "
              f"{info.absolute_zero_pct:7.1f} {info.longest_gap_ms:7.0f}ms {info.floor_db:7.1f}dB  {verdict}")

    infos = [info for _p, info in rows if isinstance(info, HardGatingInfo)]
    gated = [i for i in infos if i.is_gated]
    print(f"\n{len(infos)} decodable files | gated (>= {args.min_gap_ms:g} ms) {len(gated)} | "
          f"severe (>= 200 ms) {sum(1 for i in gated if i.is_severe)} | clean {len(infos) - len(gated)}")

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as handle:
            handle.write("file\tduration_s\tgated_pct\tabsolute_zero_pct\tlongest_gap_ms\tfloor_dbfs\tverdict\n")
            for path, info in rows:
                if isinstance(info, HardGatingInfo):
                    verdict = "SEVERE" if info.is_severe else ("gated" if info.is_gated else "clean")
                    handle.write(f"{path}\t{info.duration:.2f}\t{info.gated_sample_pct:.1f}\t"
                                 f"{info.absolute_zero_pct:.1f}\t{info.longest_gap_ms:.0f}\t"
                                 f"{info.floor_db:.1f}\t{verdict}\n")
        print(f"wrote {args.tsv}")
    return 0


def _floored_output_path(input_path: str, out_dir: str) -> str:
    stem = Path(input_path).stem
    directory = Path(out_dir) if out_dir else Path(input_path).parent
    return str(directory / f"{stem}_floored.flac")


def _repair(sound: Sound, args: argparse.Namespace):
    """ Runs the app's repair; an unset --fade-ms uses the function's own default. """
    fn = HardGateUtil.fill_core_runs
    kwargs: dict = dict(
        floor_db=args.floor_db, min_gap_ms=args.min_gap_ms, near_zero_amplitude=args.near_zero,
        room_tone=args.room_tone, match_level=args.match_level,
    )
    if args.fade_ms is not None:
        kwargs["fade_ms"] = args.fade_ms
    return fn(sound, **kwargs)


def cmd_floor(args: argparse.Namespace) -> int:
    sound = load_sound(args.input)
    if isinstance(sound, str):
        print(f"Could not load {args.input}: {sound}")
        return 1
    before = inspect_gating(sound, args)
    print(f"{args.input}")
    print_info("before", before)
    if not before.is_gated and not args.force:
        print("Nothing to repair (no hard-gated runs); pass --force to write a copy anyway.")
        return 0

    repaired, report = _repair(sound, args)
    out_path = args.out_path or _floored_output_path(args.input, args.out_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    error = SoundFileUtil.save_flac(repaired, out_path)
    if error:
        print(f"Could not save {out_path}: {error}")
        return 1
    after = inspect_gating(repaired, args)
    print_info("after ", after)
    print(f"filled {report.regions_filled} run(s), {report.filled_ms:.0f} ms total at "
          f"{report.target_db:.1f} dBFS ({report.level_source} level, "
          f"{report.spectrum_source} spectrum)")
    print(f"wrote {out_path}")
    return 0


def _coerce_option(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _make_voice_ref(sound: Sound) -> str:
    data = np.asarray(sound.data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    buffer = io.BytesIO()
    soundfile.write(buffer, data, sound.sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _resolve_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.texts:
        with open(args.texts, encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]
    elif args.project:
        project_text = Path(args.project) / "project_text.json"
        book = json.loads(project_text.read_text(encoding="utf-8"))["book"]
        groups = [g for section in book["sections"] for g in section["phrase_groups"]]
        start, _, end = args.lines.partition("-")
        first = int(start)
        last = int(end) if end else first
        for line in range(first, last + 1):
            texts.append("".join(p["text"] for p in groups[line - 1]["phrases"]).strip())
    else:
        texts = list(DEFAULT_TEXTS)
    if args.limit:
        texts = texts[: args.limit]
    return [t for t in texts if t]


def _seed_for(text: str, seed_index: int) -> int:
    digest = hashlib.sha256(f"{text}:{seed_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _build_payload(args: argparse.Namespace, voice_uri: str, text: str, seed: int) -> dict:
    payload: dict = {
        "model": args.model,
        "input": text,
        "response_format": "wav",
        "seed": seed,
        "voice_ref": {"type": "base64", "data": voice_uri},
    }
    if args.language:
        payload["language"] = args.language
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None and args.top_k > 0:
        payload["top_k"] = args.top_k
    for item in args.param or []:
        key, _, value = item.partition("=")
        payload[key] = _coerce_option(value)
    options: dict[str, int | float | str] = {}
    for item in ([] if args.no_pinned_options else ["text_chunk_size=100000"]):
        key, _, value = item.partition("=")
        options[key] = _coerce_option(value)
    for item in args.option or []:
        key, _, value = item.partition("=")
        options[key] = _coerce_option(value)
    if options:
        payload["options"] = options
    reference_text = args.reference_text
    if args.reference_text_file:
        reference_text = Path(args.reference_text_file).read_text(encoding="utf-8").strip()
    if reference_text:
        payload["reference_text"] = reference_text
    return payload


def _output_floor_db(path: str) -> float:
    data, sr = soundfile.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    window = max(1, round(0.02 * sr))
    frames = data[: (len(data) // window) * window].reshape(-1, window).astype(np.float64)
    if frames.size == 0:
        return float("-inf")
    db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    return float(np.percentile(db, 10))


def cmd_ab(args: argparse.Namespace) -> int:
    import httpx

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or os.path.join(tempfile.gettempdir(), "reference_gating_ab", stamp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ref_a = load_sound(args.ref_a)
    if isinstance(ref_a, str):
        print(f"Could not load {args.ref_a}: {ref_a}")
        return 1
    if args.ref_b:
        ref_b = load_sound(args.ref_b)
        if isinstance(ref_b, str):
            print(f"Could not load {args.ref_b}: {ref_b}")
            return 1
        ref_b_path = args.ref_b
    else:
        ref_b, report = _repair(ref_a, args)
        ref_b_path = os.path.join(out_dir, Path(args.ref_a).stem + "_floored.flac")
        error = SoundFileUtil.save_flac(ref_b, ref_b_path)
        if error:
            print(f"Could not save floored reference: {error}")
            return 1
        print(f"generated floored reference ({report.regions_filled} run(s), "
              f"{report.filled_ms:.0f} ms at {report.target_db:.1f} dBFS): {ref_b_path}")

    print("references:")
    print_info("A", inspect_gating(ref_a, args))
    print_info("B", inspect_gating(ref_b, args))

    texts = _resolve_texts(args)
    if not texts:
        print("No texts to run.")
        return 1
    url = args.server.rstrip("/") + "/v1/audio/speech"
    payload_a = _build_payload(args, _make_voice_ref(ref_a), texts[0], 0)
    shown = json.loads(json.dumps({k: v for k, v in payload_a.items() if k != "voice_ref"}))
    shown["voice_ref"] = f"<base64 wav, {len(payload_a['voice_ref']['data'])} chars>"
    shown["seed"] = "<derived from text + seed index>"
    print(f"\nPOST {url}\n{json.dumps(shown, indent=2)}\n"
          f"{len(texts)} text(s) x {args.seeds} seed(s) x 2 arms\n")

    want_wer = args.wer
    asr = None
    if want_wer:
        try:
            from faster_whisper import WhisperModel

            asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        except Exception as exception:  # noqa: BLE001
            print(f"WER disabled ({type(exception).__name__}: {exception})")
            want_wer = False

    timeout = httpx.Timeout(connect=5.0, write=60.0, read=args.timeout, pool=5.0)
    results: list[dict] = []
    voice_uris = {"A": _make_voice_ref(ref_a), "B": _make_voice_ref(ref_b)}
    with httpx.Client(timeout=timeout) as client:
        for text_index, text in enumerate(texts):
            for seed_index in range(args.seeds):
                seed = _seed_for(text, seed_index)
                row: dict = {"text_index": text_index, "seed": seed, "chars": len(text),
                             "text": text[:60]}
                for arm in ("A", "B"):
                    payload = _build_payload(args, voice_uris[arm], text, seed)
                    started = time.time()
                    try:
                        response = client.post(url, json=payload, headers={"Accept": "audio/wav"})
                        elapsed = time.time() - started
                        row[f"{arm}_status"] = response.status_code
                        row[f"{arm}_elapsed"] = round(elapsed, 2)
                        if response.status_code == 200:
                            wav_path = os.path.join(out_dir, f"{arm}_{text_index}_{seed}.wav")
                            Path(wav_path).write_bytes(response.content)
                            row[f"{arm}_wav"] = wav_path
                            row[f"{arm}_floor_db"] = round(_output_floor_db(wav_path), 1)
                            if want_wer and asr is not None:
                                segments, _info = asr.transcribe(wav_path, language="en")
                                row[f"{arm}_hyp"] = "".join(s.text for s in segments).strip()
                                row[f"{arm}_wer"] = round(_wer(text, row[f"{arm}_hyp"]), 3)
                        else:
                            row[f"{arm}_error"] = _error_message(response)
                    except httpx.HTTPError as exception:
                        row[f"{arm}_status"] = 0
                        row[f"{arm}_elapsed"] = round(time.time() - started, 2)
                        row[f"{arm}_error"] = f"{type(exception).__name__}: {exception}"
                results.append(row)
                print(f"[{text_index + 1}/{len(texts)} seed {seed_index}] "
                      f"A={_cell(row, 'A')}  B={_cell(row, 'B')}  {text[:40]!r}")

    _print_ab_summary(results, want_wer)
    results_path = os.path.join(out_dir, "results.json")
    Path(results_path).write_text(json.dumps({"payload_template": shown, "results": results}, indent=2),
                                  encoding="utf-8")
    print(f"\nresults: {results_path}\nwavs:    {out_dir}")
    return 0


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"]
    except ValueError:
        pass
    return response.text[:200] or response.reason_phrase


def _cell(row: dict, arm: str) -> str:
    status = row.get(f"{arm}_status")
    elapsed = row.get(f"{arm}_elapsed", 0.0)
    if status == 200:
        return f"ok/{elapsed:.1f}s"
    return f"FAIL({status})/{elapsed:.1f}s"


def _wer(reference: str, hypothesis: str) -> float:
    def norm(value: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", value.lower())

    ref, hyp = norm(reference), norm(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = np.zeros((rows, cols), dtype=np.int32)
    distance[:, 0] = np.arange(rows)
    distance[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            distance[i, j] = min(
                distance[i - 1, j] + 1,
                distance[i, j - 1] + 1,
                distance[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    return float(distance[rows - 1, cols - 1]) / max(1, len(ref))


def _print_ab_summary(results: list[dict], want_wer: bool) -> None:
    print("\n=== summary ===")
    header = f"{'arm':3s} {'reqs':>5s} {'failed':>7s} {'runaway':>8s} {'other':>6s} " \
             f"{'mean_s':>7s} {'mean_floor':>11s}"
    if want_wer:
        header += f" {'mean_wer':>9s}"
    print(header)
    for arm in ("A", "B"):
        requests = len(results)
        failures = [r for r in results if r.get(f"{arm}_status") != 200]
        runway = [r for r in failures if "max_tokens" in (r.get(f"{arm}_error") or "")]
        elapsed = [r.get(f"{arm}_elapsed", 0.0) for r in results]
        floors = [r[f"{arm}_floor_db"] for r in results if f"{arm}_floor_db" in r]
        line = (f"{arm:3s} {requests:5d} {len(failures):7d} {len(runway):8d} "
                f"{len(failures) - len(runway):6d} {np.mean(elapsed):7.2f} "
                f"{np.mean(floors) if floors else float('nan'):10.1f}")
        if want_wer:
            wers = [r[f"{arm}_wer"] for r in results if f"{arm}_wer" in r]
            line += f" {np.mean(wers) if wers else float('nan'):9.3f}"
        print(line)
    other_messages: dict[str, int] = {}
    for row in results:
        for arm in ("A", "B"):
            message = row.get(f"{arm}_error")
            if message and "max_tokens" not in message:
                other_messages[message[:90]] = other_messages.get(message[:90], 0) + 1
    if other_messages:
        print("\nnon-runaway failures:")
        for message, count in sorted(other_messages.items(), key=lambda item: -item[1]):
            print(f"  {count}x {message}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hard-gated voice-reference harness (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_detection_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--near-zero", type=float, default=HARD_GATING_NEAR_ZERO_AMPLITUDE,
                            help="amplitude threshold treated as silence (default: %(default)s)")
        target.add_argument("--min-gap-ms", type=float, default=HARD_GATING_MIN_GAP_MS,
                            help="minimum run length to flag (default: %(default)s)")

    detect = sub.add_parser("detect", help="scan files/dirs for hard-gated silence")
    detect.add_argument("paths", nargs="+")
    detect.add_argument("--tsv", default="", help="write a TSV report here")
    add_detection_flags(detect)
    detect.set_defaults(func=cmd_detect)

    floor = sub.add_parser("floor", help="write a copy with the gated runs re-floored")
    floor.add_argument("input")
    floor.add_argument("--out-path", default="", help="explicit output file (default: <stem>_floored.flac)")
    floor.add_argument("--out-dir", default="", help="output directory (default: alongside the input)")
    floor.add_argument("--floor-db", type=float, default=-65.0, help="target noise floor (default: %(default)s)")
    floor.add_argument("--room-tone", choices=("match", "pink", "white"), default="match",
                       help="noise spectrum: match the clip's own quiet frames, or synthesized (default: %(default)s)")
    floor.add_argument("--match-level", action="store_true",
                       help="use the clip's measured room-tone level instead of --floor-db")
    floor.add_argument("--fade-ms", type=float, default=None, help="edge fade per filled run (default: %s ms)" % 3)
    floor.add_argument("--force", action="store_true", help="write a copy even when nothing is flagged")
    add_detection_flags(floor)
    floor.set_defaults(func=cmd_floor)

    ab = sub.add_parser("ab", help="compare two references on the same texts and seeds")
    ab.add_argument("--ref-a", required=True)
    ab.add_argument("--ref-b", default="", help="default: floor --ref-a into the output dir")
    ab.add_argument("--server", default="http://127.0.0.1:8080")
    ab.add_argument("--model", default="higgs-audio-tts")
    ab.add_argument("--language", default="en")
    ab.add_argument("--temperature", type=float, default=0.8)
    ab.add_argument("--top-p", type=float, default=0.8)
    ab.add_argument("--top-k", type=int, default=30, help="0 omits top_k from the request")
    ab.add_argument("--param", action="append", default=None,
                    help="extra top-level request field KEY=VALUE (repeatable), e.g. repetition_penalty=1.2")
    ab.add_argument("--option", action="append", default=None,
                    help="extra request option KEY=VALUE (repeatable)")
    ab.add_argument("--no-pinned-options", action="store_true",
                    help="do not add the app's default text_chunk_size=100000 pin")
    ab.add_argument("--reference-text", default="")
    ab.add_argument("--reference-text-file", default="")
    ab.add_argument("--texts", default="", help="file with one text per line")
    ab.add_argument("--project", default="", help="project dir to read project_text.json from")
    ab.add_argument("--lines", default="1", help="1-based line range for --project, e.g. 1661-1680")
    ab.add_argument("--limit", type=int, default=0, help="use only the first N texts")
    ab.add_argument("--seeds", type=int, default=2)
    ab.add_argument("--timeout", type=float, default=300.0)
    ab.add_argument("--wer", action="store_true", help="also score outputs against the source text")
    ab.add_argument("--out-dir", default="")
    ab.add_argument("--floor-db", type=float, default=-65.0)
    ab.add_argument("--room-tone", choices=("match", "pink", "white"), default="match")
    ab.add_argument("--match-level", action="store_true")
    ab.add_argument("--fade-ms", type=float, default=None)
    add_detection_flags(ab)
    ab.set_defaults(func=cmd_ab)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
