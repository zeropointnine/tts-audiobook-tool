"""
Room reverb A/B renders, for choosing reverb designs and levels by ear (the
`room_reverb` project setting's presets came from these; see
tts_audiobook_tool/sound/room_reverb.py and docs-dev/room-reverb.md). Uses the
app's IR generator and streaming reverb.

Renders a range of a project's segments through the real concat flow (load,
resample, high shelf, sentence-gap extension, gap limiting, pauses with
pseudo-silence compensation), once, without room tone. Then, for each variant:

    dry render -> room reverb (streamed, chunk by chunk) -> room tone -> loudness normalization

and writes one FLAC per variant, all level-matched by the project's loudness
normalization, plus each variant's impulse response (IR) as a WAV.

The reverb is plain convolution with a synthetic IR: pre-delay, a cluster of
early reflections (each smeared into a short noise burst, to limit mono comb
filtering), and an exponentially decaying late tail with faster high-frequency
decay; the whole IR is band-limited (reverb EQ). Level is the
direct-to-reverberant ratio (DRR): dry energy over IR energy, in dB, so a
higher number is drier.

Variants are presets (see PRESETS): the small room at its subtle levels, then
bigger rooms rather than more of the small one. An external IR (`--ir`, e.g. a
recorded treated room) can be rendered alongside.

The reverb is streamed in segment-sized chunks with its tail carried across
calls, the same way the app would apply it, so tails decay naturally across
segment boundaries and pauses.

`--stereo` renders the app's RoomReverb presets with stereo reverb instead:
the voice stays mono and centered, only the reverb differs between the ears
(see "Stereo" below), as mono / late tail spread at two widths / early and late
spread. Every file is written as stereo, mono variants with identical
channels: loudness measurement counts a stereo file's channels together, so a
mono file normalized to the same target would play 3 dB louder in each ear.

Limitation: the dry render already contains any break sound effects, so here
they'd be reverberated too (the app keeps them dry). Pick a range without
breaks.

Usage:
    ./venv-base/bin/python -m testx.room_reverb_ab
    ./venv-base/bin/python -m testx.room_reverb_ab --only hall,cathedral --drr-offset 3
    ./venv-base/bin/python -m testx.room_reverb_ab --ir /path/to/room_ir.wav --ir-drrs 22,17
    ./venv-base/bin/python -m testx.room_reverb_ab --stereo
    ./venv-base/bin/python -m testx.room_reverb_ab --stereo --only "large room"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys

import numpy as np
import soundfile
import soxr

from tts_audiobook_tool.app_types import NormalizationType, RoomReverb, RoomTone, Sound
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.sound.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.sound.noise_bed import NoiseBed
from tts_audiobook_tool.sound.room_reverb import (
    ROOM_DESIGNS, RoomIrParams, StreamingReverb, finish_ir, make_room_ir, make_stereo_irs,
)


DEFAULT_PROJECT_DIR = "/d/w/w/slowgods"
DEFAULT_START = 1025  # 1-based, as in segment file names
DEFAULT_END = 1040


@dataclass(frozen=True)
class ReverbPreset:
    name: str
    room: RoomIrParams
    drr_db: float


# Presets rather than one room at increasing levels. Energy in the first
# ~50 ms fuses with the direct sound and is heard as timbre, not space: in mono
# it adds a comb-like spectral ripple that grows with level (the "phasey",
# robotic quality of a small room pushed hard). So a small room only works
# while subtle; more reverb comes from bigger rooms, whose energy arrives later
# and more diffusely (lower ripple at the same DRR), at a higher DRR because
# their long tails are far more audible in pauses. Bigger rooms also get a
# darker, thinner reverb EQ so the tail doesn't muddy the speech.
SMALL_ROOM = ROOM_DESIGNS["small"]
MEDIUM_ROOM = ROOM_DESIGNS["medium"]
# Candidates that weren't adopted, kept for comparison
HALL = replace(
    SMALL_ROOM, rt60=1.6, predelay_ms=25.0, er_count=24, er_window_ms=90.0, er_burst_ms=4.0,
    late_db=6.0, late_onset_ms=15.0, late_ramp_ms=60.0, hf_rt60_factor=0.5,
    highpass_hz=300.0, lowpass_hz=5500.0,
)
CATHEDRAL = replace(
    SMALL_ROOM, rt60=2.8, predelay_ms=35.0, er_count=20, er_window_ms=120.0, er_burst_ms=5.0,
    late_db=9.0, late_onset_ms=20.0, late_ramp_ms=80.0, hf_rt60_factor=0.45,
    highpass_hz=300.0, lowpass_hz=5000.0,
)

# Round 2 of listening tests: subtle, light and moderate-a were adopted as the
# app's RoomReverb presets "Small room, light", "Small room, moderate" and "Larger room".
PRESETS: list[ReverbPreset] = [
    ReverbPreset("subtle", SMALL_ROOM, 22.0),
    ReverbPreset("light", SMALL_ROOM, 17.0),
    ReverbPreset("moderate-a medium-room", MEDIUM_ROOM, 21.0),
    ReverbPreset("moderate-b medium-room", MEDIUM_ROOM, 25.0),
    ReverbPreset("high-a hall", HALL, 24.0),
    ReverbPreset("high-b hall", HALL, 28.0),
    ReverbPreset("high-a cathedral", CATHEDRAL, 26.0),
    ReverbPreset("high-b cathedral", CATHEDRAL, 30.0),
]


def load_external_ir(path: str, p: RoomIrParams, sr: int, max_s: float = 0.8) -> np.ndarray:
    """
    Recorded IR -> unit-energy reverb-only IR: mono, resampled, trimmed to start
    at its direct-sound peak, direct sound removed, `p.predelay_ms` inserted
    in front, same reverb EQ as the synthetic IR.
    """
    data, file_sr = soundfile.read(path, dtype="float64", always_2d=True)
    data = data.mean(axis=1)
    if file_sr != sr:
        data = soxr.resample(data, file_sr, sr)
    peak = int(np.argmax(np.abs(data)))
    data = data[peak:peak + round(max_s * sr)].copy()
    # Remove the direct sound (first ~2.5 ms after the peak), easing back in
    direct_n = round(0.0025 * sr)
    ease_n = round(0.001 * sr)
    data[:direct_n] = 0
    data[direct_n:direct_n + ease_n] *= np.linspace(0, 1, ease_n)
    # Re-align so the first reflection lands at the pre-delay
    data = np.concatenate([np.zeros(max(0, round(p.predelay_ms / 1000 * sr) - direct_n)), data])
    return finish_ir(data, p, sr)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_dry(project_dir: str, start: int, end: int, out_dir: Path) -> tuple[np.ndarray, list[int], RoomTone, NormalizationType] | str:
    """
    Real concat flow over 1-based segments start..end, without room tone.
    Returns (samples, per-segment chunk lengths, project room tone, project normalization).
    """
    project = ProjectLoadUtil.load_using_dir_path(
        project_dir, prompt_on_warnings=False, prompt_on_migration=False)
    if isinstance(project, str):
        return project

    index_start, index_end = start - 1, end - 1
    num_missing = ConcatUtil.num_missing_in(project, index_start, index_end)
    if num_missing:
        print(f"Warning: {num_missing} segment(s) in range have no audio")

    dry_path = str(out_dir / "_dry_render.flac")
    result = ConcatUtil.concatenate_sound_segments(
        dry_path,
        ConcatUtil.make_phrases_and_paths(project, index_start, index_end),
        use_break_sound_effect=project.use_break_sound_effect,
        high_shelf=project.get_high_shelf(),
        reason_pauses=project.reason_pauses,
        print_progress=True,
        section_start_indices=ProjectBookUtil.get_section_start_indices(project),
        start_group_index=index_start,
        limit_silence_gaps=project.limit_silence_gaps,
        limit_silence_gaps_duration=project.limit_silence_gaps_duration,
        min_sentence_silence=project.min_segment_sentence_silence.duration,
        room_tone_db=None,
    )
    if isinstance(result, str):
        return result

    data, sr = soundfile.read(dry_path, dtype="float64")
    os.remove(dry_path)
    if sr != APP_SAMPLE_RATE:
        return f"Unexpected sample rate {sr}"
    chunks = [round(d * sr) for d in result.durations if d > 0]
    if sum(chunks) != len(data):
        return f"Chunk lengths ({sum(chunks)}) don't add up to the render ({len(data)})"
    return data, chunks, project.room_tone, project.normalization_type


def render_variant(
        dry: np.ndarray,
        chunks: list[int],
        ir_unit: np.ndarray | None,
        drr_db: float,
        room_tone_db: float | None,
) -> np.ndarray:
    """ Reverb, then room tone, both streamed chunk by chunk as in concat """
    sr = APP_SAMPLE_RATE
    reverb = StreamingReverb(ir_unit, drr_db) if ir_unit is not None else None
    bed = NoiseBed(sr, room_tone_db) if room_tone_db is not None else None
    out: list[np.ndarray] = []
    pos = 0
    for i, n in enumerate(chunks):
        chunk = dry[pos:pos + n]
        pos += n
        if reverb:
            chunk = reverb.process(chunk)
        if bed:
            chunk = bed.apply(Sound(chunk.astype(np.float32), sr), is_last=(i == len(chunks) - 1)).data
        out.append(np.asarray(chunk, dtype=np.float64))
    return np.concatenate(out)


def save_variant(
        data: np.ndarray, path: Path, normalization: NormalizationType | None, label: str
) -> str:
    peak = float(np.max(np.abs(data)))
    if peak > 1.0:
        if normalization is None:
            print(f"  {label}: peak {peak:.3f} > 1.0, clipping")
            data = np.clip(data, -1.0, 1.0)
        else:
            # Normalization sets the final level anyway, so scale rather than clip
            data = data / peak
    if normalization is None:
        soundfile.write(str(path), data, APP_SAMPLE_RATE, subtype="PCM_24")
        return ""
    temp_path = str(path.with_name("_prenorm_" + path.name))
    soundfile.write(temp_path, data, APP_SAMPLE_RATE, subtype="PCM_24")
    err = LoudnessNormalizationUtil.normalize_file(temp_path, normalization.value, str(path))
    os.remove(temp_path)
    return err


def _early_bands_db(ir_unit: np.ndarray, drr_db: float, sr: int) -> np.ndarray:
    """ 1/12-octave band levels (dB), 250 Hz - 6 kHz, of direct sound + first 50 ms of reverb """
    n50 = round(0.05 * sr)
    h = ir_unit[:n50] * 10 ** (-drr_db / 20)
    h[0] += 1.0
    nfft = 1 << 17
    mag = np.abs(np.fft.rfft(h, nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    centers = 250 * 2 ** np.arange(0, np.log2(6000 / 250), 1 / 24)
    bands = [
        np.sqrt(np.mean(mag[(freqs > c * 2 ** (-1 / 24)) & (freqs < c * 2 ** (1 / 24))] ** 2))
        for c in centers
    ]
    return 20 * np.log10(np.asarray(bands))


def early_ripple_db(ir_unit: np.ndarray, drr_db: float, sr: int) -> float:
    """
    Coloration estimate: spectral ripple (std dev, dB) of direct sound + the
    first 50 ms of reverb, 1/12-octave smoothed, 250 Hz - 6 kHz. That window
    fuses with the direct sound and is heard as timbre; the small room at the
    level that sounded phasey (DRR 12) measures about 0.9 dB.
    """
    return float(np.std(_early_bands_db(ir_unit, drr_db, sr)))


def two_ear_ripple_db(ir_l: np.ndarray, ir_r: np.ndarray, drr_db: float, sr: int) -> float:
    """
    Crude binaural coloration estimate: ripple of the two ears' band powers,
    averaged. When the ears get different comb patterns, hearing partly
    averages them out ("binaural decoloration"); per-ear ripple doesn't show that.
    """
    bl, br = _early_bands_db(ir_l, drr_db, sr), _early_bands_db(ir_r, drr_db, sr)
    return float(np.std(10 * np.log10((10 ** (bl / 10) + 10 ** (br / 10)) / 2)))


# ---------------------------------------------------------------------------
# Stereo (--stereo)
# ---------------------------------------------------------------------------
#
# The voice stays mono (identical in both channels, dead center); only the
# reverb differs between the ears. See room_reverb.make_stereo_irs (mid/side
# spread of the late tail, and optionally the early reflections). Keeping the
# mid part at full level instead would make the mono fold-down exact, but each
# ear wetter than the mono preset, confounding the comparison.
#
# Round 3 of listening tests: all four sounded good; late-w100 and all-w100
# were adopted as the app's RoomReverbStereo Wide and Wider.

@dataclass(frozen=True)
class StereoSpec:
    name: str
    spread_early: bool
    width: float  # 0 = mono, 1 = fully decorrelated


STEREO_SPECS: list[StereoSpec] = [
    StereoSpec("a mono", False, 0.0),
    StereoSpec("b late-w50", False, 0.5),
    StereoSpec("c late-w100", False, 1.0),
    StereoSpec("d all-w100", True, 1.0),
]

def render_stereo_variant(
        dry: np.ndarray,
        chunks: list[int],
        irs: tuple[np.ndarray, np.ndarray] | None,
        drr_db: float,
        room_tone_db: float | None,
) -> np.ndarray:
    """
    Stereo (n, 2): mono voice + per-channel reverb, then room tone (identical
    in both channels), streamed chunk by chunk as in concat
    """
    sr = APP_SAMPLE_RATE
    reverbs = (StreamingReverb(irs[0], drr_db), StreamingReverb(irs[1], drr_db)) if irs else None
    bed = NoiseBed(sr, room_tone_db) if room_tone_db is not None else None
    out: list[np.ndarray] = []
    pos = 0
    for i, n in enumerate(chunks):
        chunk = dry[pos:pos + n]
        pos += n
        if reverbs:
            stereo = np.stack([reverbs[0].process(chunk), reverbs[1].process(chunk)], axis=1)
        else:
            stereo = np.stack([chunk, chunk], axis=1).astype(np.float64)
        if bed:
            noise = bed.apply(Sound(np.zeros(n, dtype=np.float32), sr), is_last=(i == len(chunks) - 1)).data
            stereo += noise[:, None]
        out.append(stereo)
    return np.concatenate(out)


def run_stereo(
        args: argparse.Namespace,
        dry: np.ndarray,
        chunks: list[int],
        room_tone: RoomTone,
        normalization: NormalizationType | None,
) -> int:
    sr = APP_SAMPLE_RATE
    out_dir = Path(args.out_dir or os.path.join(args.project, "reverb_ab", "stereo"))
    out_dir.mkdir(parents=True, exist_ok=True)

    presets = [p for p in RoomReverb if p.design is not None and p.drr_db is not None]
    if args.only:
        keys = [k.strip().lower() for k in args.only.split(",") if k.strip()]
        presets = [p for p in presets if any(k in f"{p.label} {p.id}".lower() for k in keys)]

    tag = f"{args.start}-{args.end}"
    variants: list[tuple[str, tuple[np.ndarray, np.ndarray] | None, float]] = [("0 dry", None, 0.0)]
    print(f"{'variant':44s} {'corr':>5s} {'ear L':>6s} {'ear R':>6s} {'ripple':>7s} {'2-ear':>6s} {'mono sum':>8s}")
    for i, preset in enumerate(presets):
        assert preset.design is not None and preset.drr_db is not None
        drr = preset.drr_db + args.drr_offset
        room = ROOM_DESIGNS[preset.design]
        mono_ir = make_room_ir(room, sr)
        for spec in STEREO_SPECS:
            ir_l, ir_r = make_stereo_irs(room, sr, spec.width, spec.spread_early)
            label = f"{i + 1}{spec.name} {preset.label.lower().replace(' ', '-')} drr{drr:g}"
            variants.append((label, (ir_l, ir_r), drr))
            # Per-ear reverb energy and the mono fold-down, both relative to the mono preset
            corr = np.sum(ir_l * ir_r) / np.sqrt(np.sum(ir_l ** 2) * np.sum(ir_r ** 2))
            ear_l = 10 * np.log10(np.sum(ir_l ** 2) / np.sum(mono_ir ** 2))
            ear_r = 10 * np.log10(np.sum(ir_r ** 2) / np.sum(mono_ir ** 2))
            fold = 10 * np.log10(np.sum(((ir_l + ir_r) / 2) ** 2) / np.sum(mono_ir ** 2))
            print(f"{label:44s} {corr:5.2f} {ear_l:+6.2f} {ear_r:+6.2f} "
                  f"{early_ripple_db(ir_l, drr, sr):7.2f} {two_ear_ripple_db(ir_l, ir_r, drr, sr):6.2f} {fold:+8.2f}")
    print("(corr: reverb interchannel correlation; ear L/R and mono sum: reverb level vs the mono preset, dB;")
    print(" ripple: left ear early coloration, dB; 2-ear: crude binaural estimate, dB)")
    print()

    for label, irs, drr in variants:
        path = out_dir / f"{tag} {label}.flac"
        print(f"--- {path.name}")
        data = render_stereo_variant(dry, chunks, irs, drr, room_tone.level_db)
        err = save_variant(data, path, normalization, label)
        if err:
            print(f"Error: {err}")
            return 1

    print(f"Saved to: {out_dir}")
    return 0


def parse_drrs(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=DEFAULT_PROJECT_DIR, help="project directory")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="first segment (1-based)")
    parser.add_argument("--end", type=int, default=DEFAULT_END, help="last segment (1-based, inclusive)")
    parser.add_argument("--out-dir", default="", help="default: <project>/reverb_ab")
    parser.add_argument("--only", default="",
                        help="comma-separated substrings; render only presets whose name matches one")
    parser.add_argument("--drr-offset", type=float, default=0.0,
                        help="add to every preset's DRR (dB; positive = drier)")
    parser.add_argument("--ir", default="", help="also render a recorded IR file (small room EQ and pre-delay)")
    parser.add_argument("--ir-drrs", default="22,17", help="DRRs (dB) for --ir")
    parser.add_argument("--room-tone", choices=["project"] + [t.id for t in RoomTone], default="project")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--stereo", action="store_true",
                        help="render the app's presets with stereo reverb (mono voice); --only matches "
                             "preset labels/ids, --ir is ignored")
    args = parser.parse_args()

    L.init("room_reverb_ab")
    sr = APP_SAMPLE_RATE
    out_dir = Path(args.out_dir or os.path.join(args.project, "reverb_ab"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stereo:
        result = render_dry(args.project, args.start, args.end, out_dir)
        if isinstance(result, str):
            print(f"Error: {result}")
            return 1
        dry, chunks, project_room_tone, normalization = result
        if args.no_normalize:
            normalization = None
        room_tone = project_room_tone if args.room_tone == "project" else RoomTone.get_by_id(args.room_tone)
        assert room_tone is not None
        print(f"Rendered {len(chunks)} segments, {len(dry) / sr:.1f}s")
        print(f"Room tone: {room_tone.label}  Normalization: {normalization.value.label if normalization else 'Disabled'}")
        print()
        return run_stereo(args, dry, chunks, room_tone, normalization)

    presets = PRESETS
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        presets = [p for p in presets if any(k in p.name for k in keys)]
    presets = [replace(p, drr_db=p.drr_db + args.drr_offset) for p in presets]

    result = render_dry(args.project, args.start, args.end, out_dir)
    if isinstance(result, str):
        print(f"Error: {result}")
        return 1
    dry, chunks, project_room_tone, normalization = result
    if args.no_normalize:
        normalization = None
    room_tone = project_room_tone if args.room_tone == "project" else RoomTone.get_by_id(args.room_tone)
    assert room_tone is not None

    print(f"Rendered {len(chunks)} segments, {len(dry) / sr:.1f}s")
    print(f"Room tone: {room_tone.label}  Normalization: {normalization.value.label if normalization else 'Disabled'}")
    print()

    # One IR per distinct room
    ir_dir = out_dir / "ir"
    ir_dir.mkdir(exist_ok=True)
    room_irs: dict[RoomIrParams, np.ndarray] = {}
    for preset in presets:
        if preset.room not in room_irs:
            ir = make_room_ir(preset.room, sr)
            room_irs[preset.room] = ir
            room_name = preset.name.split()[-1] if " " in preset.name else "small-room"
            soundfile.write(str(ir_dir / f"{room_name}.wav"), ir / np.max(np.abs(ir)) * 0.9, sr, subtype="FLOAT")
            print(f"IR {room_name}: {len(ir) / sr * 1000:.0f} ms  {preset.room}")
    print()

    tag = f"{args.start}-{args.end}"
    variants: list[tuple[str, np.ndarray | None, float]] = [("0 dry", None, 0.0)]
    for i, preset in enumerate(presets):
        variants.append((f"{i + 1} {preset.name} drr{preset.drr_db:g}", room_irs[preset.room], preset.drr_db))
    if args.ir:
        ext_ir = load_external_ir(args.ir, SMALL_ROOM, sr)
        stem = Path(args.ir).stem
        for drr in parse_drrs(args.ir_drrs):
            variants.append((f"9 {stem} drr{drr:g}", ext_ir, drr))

    for label, ir, drr in variants:
        if ir is not None:
            print(f"{label:42s} early ripple {early_ripple_db(ir, drr, sr):.2f} dB")
    print()

    for label, ir, drr in variants:
        path = out_dir / f"{tag} {label}.flac"
        print(f"--- {path.name}")
        data = render_variant(dry, chunks, ir, drr, room_tone.level_db)
        err = save_variant(data, path, normalization, label)
        if err:
            print(f"Error: {err}")
            return 1

    print(f"Saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
