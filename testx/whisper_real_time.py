#!/usr/bin/env python3

"""
Manual testbed for conversation-style real-time mic transcription.

Designed to mirror the conversation feature's use of
[`RealtimeTranscriber`](tts_audiobook_tool/conversation/realtime_transcriber.py:19)
more closely than a bare transcription demo, especially for VAD tuning.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from tts_audiobook_tool.app_types import Segment
from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.constants_config import CHAT_SILENCE_THRESHOLD_CHUNKED, CHAT_SILENCE_THRESHOLD_IMMEDIATE
from tts_audiobook_tool.l import L
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.conversation.realtime_transcriber import RealtimeTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual VAD/STT testbed aligned with conversation-mode mic capture."
    )
    parser.add_argument(
        "--mode",
        choices=("immediate", "chunked"),
        default="immediate",
        help="Mirror the conversation feature's immediate or chunked silence timeout.",
    )
    parser.add_argument(
        "--debug-vad",
        action="store_true",
        help="Enable per-interval VAD diagnostics via [`L.i()`](tts_audiobook_tool/l.py:34).",
    )
    return parser.parse_args()


def make_silence_duration_s(mode: str) -> float:
    if mode == "immediate":
        return CHAT_SILENCE_THRESHOLD_IMMEDIATE
    return CHAT_SILENCE_THRESHOLD_CHUNKED


def print_effective_settings(mode: str, util: RealtimeTranscriber) -> None:
    print("  Conversation-mode proxy settings")
    print(f"  Mode                : {mode}")
    print(f"  Silence timeout     : {util.silence_duration_s:.2f}s")
    print(f"  Silence threshold   : {util.silence_threshold:.5f}")
    print(f"  Max chunk duration  : {util.max_chunk_duration_s:.2f}s")
    print(f"  Min chunk duration  : {util.min_chunk_duration_s:.2f}s")
    print(f"  Noise window        : {util.noise_window_s:.2f}s")
    print(f"  Start ratio         : {util.speech_start_noise_ratio:.2f}")
    print(f"  Silence ratio       : {util.silence_noise_ratio:.2f}")
    print(f"  Peak silence ratio  : {util.peak_silence_ratio:.2f}")
    print(f"  Pre-speech pad      : {util.pre_speech_pad_s:.2f}s")
    print(f"  Debug VAD           : {util.debug_vad}")
    print("=" * 60)


def main() -> None:
    
    args = parse_args()
    
    from tts_audiobook_tool.prefs import Prefs
    prefs = Prefs()
    
    silence_duration_s = make_silence_duration_s(args.mode)

    try:
        L.init("tts_audiobook_tool")
    except Exception:
        pass

    # ── device info ───────────────────────────────────────────────────────────
    dev = sd.query_devices(kind="input")
    assert isinstance(dev, dict), "No input device found"
    print("=" * 60)
    print(f"  Mic     : {dev['name']}")
    print(f"  Mic SR  : {dev['default_samplerate']:.0f} Hz  →  resampled to {WHISPER_SAMPLERATE} Hz")
    print(f"  Mic ch  : {dev['max_input_channels']}")
    print(f"  Model   : {prefs.stt_variant.id}")
    print(f"  Device  : {prefs.stt_config.device} / {prefs.stt_config.compute_type}")
    print("=" * 60)

    print("  Loading STT model ...")
    Stt.set_variant(prefs.stt_variant)
    Stt.set_config(prefs.stt_config)
    Stt.get_whisper()
    print("  STT model ready.")
    print("=" * 60)

    session_start = time.perf_counter()
    chunk_count = 0
    total_audio_s = 0.0
    total_words = 0
    chunk_dispatch_time: float | None = None
    chunk_samples = 0

    def on_chunk_dispatched(duration_s: float) -> None:
        nonlocal chunk_count, total_audio_s, chunk_dispatch_time
        chunk_count += 1
        total_audio_s += duration_s
        chunk_dispatch_time = time.perf_counter()
        session_s = chunk_dispatch_time - session_start
        print(
            f"\n[{session_s:6.1f}s]  chunk #{chunk_count}"
            f"  |  {duration_s:.2f}s audio  →  Whisper ..."
        )

    def on_transcription(segments: list[Segment], audio: np.ndarray | None = None) -> None:
        nonlocal total_words, chunk_dispatch_time, chunk_samples
        now = time.perf_counter()
        inference_s = (now - chunk_dispatch_time) if chunk_dispatch_time is not None else 0.0

        words = [w for seg in segments for w in (seg.words or [])]
        total_words += len(words)
        avg_prob = sum(w.probability for w in words) / len(words) if words else 0.0
        text = " ".join(seg.text.strip() for seg in segments)
        chunk_samples = int(audio.size) if audio is not None else 0
        chunk_audio_s = chunk_samples / WHISPER_SAMPLERATE if chunk_samples else 0.0

        print(
            f"           ↳ {inference_s:.2f}s inference"
            f"  |  {len(segments)} seg  {len(words)} words"
            + (f"  avg_prob {avg_prob:.2f}" if words else "")
        )
        if chunk_samples:
            print(
                f"             prepared audio: {chunk_samples} samples"
                f"  ({chunk_audio_s:.2f}s @ {WHISPER_SAMPLERATE} Hz)"
            )
        print(f"           \"{text}\"")

        if words:
            first, last = words[0], words[-1]
            print(
                f"             word timing: [{first.word.strip()} @{first.start:.2f}s]"
                f" … [{last.word.strip()} @{last.end:.2f}s]"
            )

    util = RealtimeTranscriber(
        prefs=prefs,
        on_transcription=on_transcription,
        silence_duration_s=silence_duration_s,
        on_chunk_dispatched=on_chunk_dispatched,
        debug_vad=args.debug_vad,
    )

    print_effective_settings(args.mode, util)

    print("\nListening ...  (Ctrl+C to stop)\n")
    util.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        util.stop()
        elapsed = time.perf_counter() - session_start
        print("\n" + "=" * 60)
        print(f"  Session : {elapsed:.1f}s")
        print(f"  Chunks  : {chunk_count}")
        print(f"  Audio   : {total_audio_s:.1f}s transcribed")
        print(f"  Words   : {total_words}")
        print("=" * 60)


if __name__ == "__main__":
    main()
