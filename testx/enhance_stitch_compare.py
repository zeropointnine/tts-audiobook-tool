"""
Compare the enhance-flow transcript stitchers on real Whisper output.

Transcribes a slice of an audiobook in the windows the enhance flow uses
(30 s windows sharing 5 s), keeps the raw per-window word lists, and stitches
them both ways: the original timestamp-threshold stitcher and the overlap
stitcher (`ENHANCE_OVERLAP_STITCHING`). Reports per stitcher:

- word count and adjacent duplicate words at seams,
- when the enhance book (`<stem>.abr.json`) sits next to the audio: source
  tokens matched and transcript tokens left unaligned when the stitched
  transcript is aligned against the book (anchored aligner); a duplicated word
  is unaligned, a dropped word costs a match,

then prints the seams where the two stitched outputs differ.

Raw windows can be saved and reloaded, to iterate on stitching without
re-running Whisper. Uses the STT model selected in the app's preferences.

Usage:
    ./venv-base/bin/python -m testx.enhance_stitch_compare book.m4b --start 3600 --minutes 10 --save-chunks /tmp/w.pkl
    ./venv-base/bin/python -m testx.enhance_stitch_compare book.m4b --load-chunks /tmp/w.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_audiobook_tool.app_types import Word
from tts_audiobook_tool.enhance import enhance_alignment
from tts_audiobook_tool.enhance import enhance_anchored_alignment as anchored
from tts_audiobook_tool.enhance.enhance_alignment import normalize_text

STRIDE = enhance_alignment.TRANSCRIBE_CHUNK_SECONDS - enhance_alignment.TRANSCRIBE_OVERLAP_SECONDS
OVERLAP = enhance_alignment.TRANSCRIBE_OVERLAP_SECONDS


def transcribe_slice(audio_path: str, start: float, minutes: float, language: str) -> list[list[Word]]:
    import soundfile
    from tts_audiobook_tool.model_worker import ModelWorker
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.sound.audio_decoder import read_audio

    samples, sample_rate = read_audio(audio_path, 16000, 1, start, minutes * 60)
    with tempfile.TemporaryDirectory() as folder:
        slice_path = str(Path(folder) / "slice.wav")
        soundfile.write(slice_path, samples, sample_rate, subtype="FLOAT")
        try:
            chunks = enhance_alignment._transcribe_stream_with_overlap(
                slice_path, Prefs.load(save_if_dirty=False), language
            )
        finally:
            ModelWorker.shutdown()
    if chunks is None:
        raise SystemExit("Interrupted")
    return chunks


def seam_duplicates(words: list[Word], window_count: int) -> int:
    count = 0
    for i in range(len(words) - 1):
        key = normalize_text(words[i].word)
        if key and key == normalize_text(words[i + 1].word):
            seam = round(words[i + 1].start / STRIDE)
            if 1 <= seam < window_count and abs(words[i + 1].start - (seam * STRIDE + OVERLAP / 2)) <= OVERLAP:
                count += 1
    return count


def book_alignment_stats(audio_path: str, words: list[Word], language: str) -> tuple[int, int, int] | None:
    """(matched source tokens, unaligned transcript tokens, transcript tokens) or None without a book."""
    from tts_audiobook_tool.enhance import enhance_text
    from tts_audiobook_tool.enhance.enhance_artifacts import EnhanceArtifacts, load_book

    book, _ = load_book(EnhanceArtifacts.from_audio_path(audio_path))
    if book is None:
        return None
    source = [t for p in enhance_text.flatten_book(book).phrases for t in anchored.tokenize(p.text, language)]
    transcript = [t for w in words for t in anchored.tokenize(w.word, language)]
    alignment = anchored._align_tokens(source, transcript, phonetic=(language == "en"))
    assert alignment is not None
    matched = sum(1 for f, c in zip(alignment.first, alignment.cost) if f >= 0 and c < 1.0)
    used: set[int] = set()
    for first, last in zip(alignment.first, alignment.last):
        if first >= 0:
            used.update(range(first, last + 1))
    return matched, len(transcript) - len(used), len(transcript)


def words_text(words: list[Word], start: float, end: float) -> str:
    return " ".join(w.word.strip() for w in words if start <= w.start < end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="audiobook file (a sibling <stem>.abr.json enables book alignment stats)")
    parser.add_argument("--start", type=float, default=0.0, help="slice start, seconds")
    parser.add_argument("--minutes", type=float, default=10.0, help="slice length, minutes")
    parser.add_argument("--language", default="en")
    parser.add_argument("--save-chunks", help="pickle the raw per-window word lists here")
    parser.add_argument("--load-chunks", help="reuse raw per-window word lists instead of transcribing")
    parser.add_argument("--show", type=int, default=12, help="differing seams to print")
    args = parser.parse_args()

    if args.load_chunks:
        with open(args.load_chunks, "rb") as file:
            chunks = pickle.load(file)
    else:
        chunks = transcribe_slice(args.audio, args.start, args.minutes, args.language)
        if args.save_chunks:
            with open(args.save_chunks, "wb") as file:
                pickle.dump(chunks, file)

    stitched = {
        "timestamp (old)": enhance_alignment._stitch_transcripts(chunks),
        "overlap (new)": enhance_alignment._stitch_transcripts_by_overlap(chunks, STRIDE, OVERLAP),
    }
    print(f"\n{len(chunks)} windows")
    for name, words in stitched.items():
        line = f"{name:16} words={len(words):6}  seam duplicates={seam_duplicates(words, len(chunks)):4}"
        stats = book_alignment_stats(args.audio, words, args.language)
        if stats:
            matched, unaligned, total = stats
            line += f"  book-matched tokens={matched:6}  unaligned transcript tokens={unaligned:5}/{total}"
        print(line)

    old, new = stitched.values()
    shown = 0
    print("\nSeams where the outputs differ (times relative to the slice):")
    for seam in range(1, len(chunks)):
        start, end = seam * STRIDE - 3, seam * STRIDE + OVERLAP + 3
        old_text, new_text = words_text(old, start, end), words_text(new, start, end)
        if old_text != new_text and shown < args.show:
            shown += 1
            print(f"  seam {seam} ({seam * STRIDE}s):")
            print(f"    old: {old_text}")
            print(f"    new: {new_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
