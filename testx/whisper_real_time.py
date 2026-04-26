#!/usr/bin/env python3

"""
Manual test: real-time mic transcription via WhisperRealTimeUtil.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
from faster_whisper.transcribe import Segment

from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.whisper_realtime_util import WhisperRealTimeUtil

DEBUG_VAD = True


def main() -> None:
    prefs = Prefs()

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

    def on_transcription(segments: list[Segment]) -> None:
        nonlocal total_words, chunk_dispatch_time
        now = time.perf_counter()
        inference_s = (now - chunk_dispatch_time) if chunk_dispatch_time is not None else 0.0

        words = [w for seg in segments for w in (seg.words or [])]
        total_words += len(words)
        avg_prob = sum(w.probability for w in words) / len(words) if words else 0.0
        text = " ".join(seg.text.strip() for seg in segments)

        print(
            f"           ↳ {inference_s:.2f}s inference"
            f"  |  {len(segments)} seg  {len(words)} words"
            + (f"  avg_prob {avg_prob:.2f}" if words else "")
        )
        print(f"           \"{text}\"")

        if words:
            first, last = words[0], words[-1]
            print(
                f"             word timing: [{first.word.strip()} @{first.start:.2f}s]"
                f" … [{last.word.strip()} @{last.end:.2f}s]"
            )

    util = WhisperRealTimeUtil(
        prefs=prefs,
        on_transcription=on_transcription,
        on_chunk_dispatched=on_chunk_dispatched,
        debug_vad=DEBUG_VAD,
    )

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
