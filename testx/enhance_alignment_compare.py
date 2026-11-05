"""
A/B comparison of the enhance-flow forced-alignment algorithms on real data.

Loads the enhance work files next to an audiobook (`<stem>.abr.json` book and
`<stem>.transcription.bin`), aligns them with both algorithms through
`enhance_text.align_book` (toggling `ENHANCE_ANCHORED_ALIGNMENT`), and reports
runtime, unmatched ("orphan") lines, attached interjections, where the two
agree, and samples of lines they place far apart, with the transcript text at
each placement. Nothing is written.

--transcription aligns against another transcription file instead of the
sibling one, eg one made with a different stitcher or STT model. Given only
--transcription, the book is the `<stem>.abr.json` next to that
`<stem>.transcription.bin`. The greedy algorithm takes minutes on a full book;
--no-greedy reports the anchored algorithm alone.

Usage:
    ./venv-base/bin/python -m testx.enhance_alignment_compare "path/to/book.m4b" [--transcription other.transcription.bin] [--show 10] [--no-greedy]
    ./venv-base/bin/python -m testx.enhance_alignment_compare --transcription "path/to/book.transcription.bin" [--no-greedy]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_audiobook_tool import constants_config
from tts_audiobook_tool.app_types import Word
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.enhance import enhance_anchored_alignment as anchored
from tts_audiobook_tool.enhance import enhance_text
from tts_audiobook_tool.enhance.enhance_alignment import normalize_text
from tts_audiobook_tool.enhance.enhance_artifacts import (
    EnhanceArtifacts,
    count_misalignments,
    load_book,
    load_transcription,
)

AGREE_SECONDS = 0.3
FAR_APART_SECONDS = 2.0
TRANSCRIPTION_SUFFIX = ".transcription.bin"


def make_artifacts(audio: str | None, transcription: str | None) -> EnhanceArtifacts:
    if audio:
        artifacts = EnhanceArtifacts.from_audio_path(audio)
    else:
        path = Path(transcription or "")
        if not path.name.endswith(TRANSCRIPTION_SUFFIX):
            raise SystemExit(
                f"Without an audio path, --transcription must be named <stem>{TRANSCRIPTION_SUFFIX} "
                "so its <stem>.abr.json book can be found"
            )
        stem = path.name[: -len(TRANSCRIPTION_SUFFIX)]
        # Only the stem matters; the placeholder suffix keeps dots in the stem intact
        artifacts = EnhanceArtifacts.from_audio_path(path.with_name(f"{stem}.audio"))
    if transcription:
        artifacts = dataclasses.replace(artifacts, transcription_path=Path(transcription))
    return artifacts


def run(book, words: list[Word], use_anchored: bool) -> tuple[list[TimedPhrase], float]:
    constants_config.ENHANCE_ANCHORED_ALIGNMENT = use_anchored
    started = time.monotonic()
    # The greedy aligner's own "Aligning N/M" progress line stays visible,
    # since it runs for minutes; the anchored one's summary line is hidden
    output = contextlib.redirect_stdout(io.StringIO()) if use_anchored else contextlib.nullcontext()
    with output:
        timed, interrupted = enhance_text.align_book(book, words)
    if interrupted:
        raise SystemExit("Interrupted")
    return timed, time.monotonic() - started


def is_unmatched(item: TimedPhrase) -> bool:
    return item.time_start == 0.0 and item.time_end == 0.0


def heard(words: list[Word], item: TimedPhrase) -> str:
    text = " ".join(w.word for w in words if item.time_start <= w.start < item.time_end)
    return normalize_text(text)[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "audio", nargs="?",
        help="source audiobook whose enhance work files sit next to it (optional with --transcription)",
    )
    parser.add_argument(
        "--transcription",
        help="transcription .bin to align against (default: <stem>.transcription.bin next to the audio)",
    )
    parser.add_argument("--show", type=int, default=10, help="far-apart lines to print")
    parser.add_argument("--no-greedy", action="store_true", help="skip the (slow) greedy algorithm")
    args = parser.parse_args()
    if not args.audio and not args.transcription:
        parser.error("give an audio path, --transcription, or both")

    artifacts = make_artifacts(args.audio, args.transcription)
    book, error = load_book(artifacts)
    if book is None:
        raise SystemExit(f"{error}: {artifacts.book_path}")
    words, error = load_transcription(artifacts)
    if words is None:
        raise SystemExit(f"{error}: {artifacts.transcription_path}")
    phrases = enhance_text.flatten_book(book).phrases
    print(f"book:          {artifacts.book_path}")
    print(f"transcription: {artifacts.transcription_path}")
    print(f"{len(phrases)} lines, {len(words)} transcript words")

    new, new_seconds = run(book, words, use_anchored=True)
    detail = anchored.align_phrases_detailed(phrases, words, book.segmentation_settings.language_code)
    statuses = Counter(item.status for item in detail.phrase_alignments)
    print(
        f"anchored: {new_seconds:7.1f}s  orphans={count_misalignments(new)}  "
        f"attached interjections={statuses[anchored.STATUS_ATTACHED]}"
    )
    if args.no_greedy:
        return 0

    print("\ngreedy: aligning (minutes on a full book; Ctrl-C to stop)...")
    old, old_seconds = run(book, words, use_anchored=False)
    print(f"greedy:   {old_seconds:7.1f}s  orphans={count_misalignments(old)}")

    both = agree = old_only = new_only = 0
    far: list[int] = []
    for index, (o, n) in enumerate(zip(old, new)):
        if is_unmatched(o) or is_unmatched(n):
            old_only += not is_unmatched(o) and is_unmatched(n)
            new_only += is_unmatched(o) and not is_unmatched(n)
            continue
        both += 1
        difference = abs(o.time_start - n.time_start)
        agree += difference < AGREE_SECONDS
        if difference >= FAR_APART_SECONDS:
            far.append(index)
    print(
        f"timed by both={both} (start within {AGREE_SECONDS}s: {agree}), "
        f"greedy only={old_only}, anchored only={new_only}, "
        f"{FAR_APART_SECONDS}s+ apart={len(far)}"
    )

    print(f"\nLines placed {FAR_APART_SECONDS}s+ apart (sampled):")
    for index in far[:: max(1, len(far) // max(1, args.show))][: args.show]:
        o, n = old[index], new[index]
        print(f"  [{index}] {phrases[index].text.strip()[:70]!r}")
        print(f"      greedy   {o.time_start:9.2f}s: {heard(words, o)!r}")
        print(f"      anchored {n.time_start:9.2f}s: {heard(words, n)!r}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
