#!/usr/bin/env python3
"""
Build a normalized Spanish whitelist using the Python `wordfreq` package.

Purpose
-------
Create a cleaner Spanish whitelist for Whisper/TTS validation without relying on
raw OpenSLR/Gigaword scrape artifacts.

`wordfreq` provides frequency-ranked word lists for many languages, including
Spanish. This script:

  1. Reads Spanish words from wordfreq in frequency order.
  2. Normalizes them using the same intended Whisper/TTS comparison form:
       - casefold/lowercase
       - strip vowel accents and ü diaeresis
       - preserve ñ
  3. Rejects malformed tokens.
  4. Merges duplicate normalized forms.
  5. Ranks by wordfreq Zipf frequency.
  6. Keeps the top N normalized tokens.
  7. Alphabetizes the final output.
  8. Writes audit files.

Install
-------
  python -m pip install wordfreq

Recommended first run
---------------------
  python build_wordfreq_spanish_whitelist_v2.py --limit 60000

Try a larger list
-----------------
  python build_wordfreq_spanish_whitelist_v2.py --limit 75000

Outputs
-------
  spanish_whitelist.txt
      Final normalized whitelist, one token per line, alphabetically sorted.

  spanish_whitelist_ranked.tsv
      Retained/available normalized candidates in frequency-rank order.

  spanish_whitelist_rejected.tsv
      Rejected raw wordfreq tokens and rejection reasons.

Notes
-----
This script keeps normal word-like proper names if wordfreq includes them.
It rejects punctuation, digits, URLs, emojis, multiword phrases, and tokens that do
not normalize to lowercase Spanish letters plus ñ.

The final whitelist is pre-normalized. Runtime source text and Whisper transcripts
should be normalized using the same policy before comparison and whitelist lookup.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


NORMALIZED_SPANISH_TOKEN_RE = re.compile(r"^[a-zñ]+$")

HAS_DIGIT_RE = re.compile(r"\d")
URL_OR_EMAILISH_RE = re.compile(r"(https?://|www\.|@)", re.IGNORECASE)
# Strict Roman numeral recognizer. This avoids the bad broad rule that would
# incorrectly reject normal Spanish words such as "mi", "mil", and "civil".
STRICT_ROMAN_NUMERAL_RE = re.compile(
    r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)

# This is intentionally small. Add project-specific junk to --stoplist instead.
BUILTIN_REJECTS = {
    "",
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "html",
    "pdf",
    "xml",
    "jpg",
    "jpeg",
    "png",
    "gif",
}


@dataclass(frozen=True)
class Candidate:
    word: str
    score: float
    raw_examples: tuple[str, ...]


def strip_spanish_diacritics_keep_enye(text: str) -> str:
    """
    Strip Spanish vowel accents/diaeresis but preserve ñ.

    Examples:
      María    -> Maria
      pingüino -> pinguino
      año      -> año
    """
    text = text.replace("ñ", "\0ENYE_LOWER\0")
    text = text.replace("Ñ", "\0ENYE_UPPER\0")

    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text)

    text = text.replace("\0ENYE_LOWER\0", "ñ")
    text = text.replace("\0ENYE_UPPER\0", "Ñ")
    return text


def normalize_spanish_token(raw: str) -> str:
    """
    Normalize one token into the final whitelist comparison form.
    """
    token = raw.strip().casefold()
    token = strip_spanish_diacritics_keep_enye(token)
    token = token.casefold()
    return token


def raw_looks_like_acronym_or_initialism(raw: str) -> bool:
    """
    Detect all-caps alphabetic tokens such as ONU, PSOE, FBI, EEUU.

    This checks the raw spelling before normalization. It does not reject simple
    titlecase names.
    """
    letters = [ch for ch in raw if ch.isalpha()]
    if len(letters) < 2:
        return False
    return all(ch.upper() == ch and ch.lower() != ch for ch in letters)


def raw_has_internal_camelcase(raw: str) -> bool:
    """
    Detect glued mixed-case tokens such as CanadaCuba, BryanMike, AlIttihad.

    This should rarely matter for wordfreq, but it is cheap insurance.
    It does not reject simple titlecase names.
    """
    letters = [ch for ch in raw if ch.isalpha()]
    if len(letters) < 3:
        return False

    has_lower = any(ch.lower() == ch and ch.upper() != ch for ch in letters)
    has_internal_upper = any(
        i > 0 and ch.upper() == ch and ch.lower() != ch
        for i, ch in enumerate(letters)
    )
    return has_lower and has_internal_upper


def should_keep(
    raw: str,
    normalized: str,
    *,
    min_len: int,
    max_len: int,
    reject_acronyms: bool,
    reject_roman_numerals: bool,
    keep_internal_camelcase: bool,
    stoplist: set[str],
) -> tuple[bool, str]:
    """
    Return (keep, reason).
    """
    if not normalized:
        return False, "empty"

    if normalized in BUILTIN_REJECTS or normalized in stoplist:
        return False, "stoplist"

    if len(normalized) < min_len:
        return False, "too_short"

    if len(normalized) > max_len:
        return False, "too_long"

    if URL_OR_EMAILISH_RE.search(raw) or normalized.startswith(("http", "www")):
        return False, "url_or_email"

    if HAS_DIGIT_RE.search(raw):
        return False, "contains_digit"

    if reject_acronyms and raw_looks_like_acronym_or_initialism(raw):
        return False, "all_caps_acronym"

    if not keep_internal_camelcase and raw_has_internal_camelcase(raw):
        return False, "internal_camelcase_glued_token"

    if not NORMALIZED_SPANISH_TOKEN_RE.fullmatch(normalized):
        return False, "malformed_after_normalization"

    if reject_roman_numerals and STRICT_ROMAN_NUMERAL_RE.fullmatch(normalized):
        return False, "roman_numeral"

    return True, "keep"


def load_stoplist(path: Path | None) -> set[str]:
    """
    Load optional custom reject list. Entries are normalized before matching.
    """
    if path is None:
        return set()

    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(normalize_spanish_token(line))
    return out


def iter_wordfreq_words(language: str, wordlist: str, max_raw_words: int | None):
    """
    Yield raw words from wordfreq.

    Import is inside this function so the script can show a clean install message
    if wordfreq is missing.
    """
    try:
        from wordfreq import iter_wordlist  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wordfreq\n\n"
            "Install it with:\n"
            "  python -m pip install wordfreq\n"
        ) from exc

    count = 0
    for raw in iter_wordlist(language, wordlist=wordlist):
        yield raw
        count += 1
        if max_raw_words is not None and count >= max_raw_words:
            break


def zipf_score(raw: str, language: str, wordlist: str) -> float:
    try:
        from wordfreq import zipf_frequency # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wordfreq\n\n"
            "Install it with:\n"
            "  python -m pip install wordfreq\n"
        ) from exc

    return float(zipf_frequency(raw, language, wordlist=wordlist))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a normalized Spanish whitelist using wordfreq."
    )
    parser.add_argument("--language", default="es", help="wordfreq language code.")
    parser.add_argument(
        "--wordlist",
        default="best",
        help=(
            "wordfreq wordlist tier. Usually use 'best'. "
            "Depending on installed wordfreq data, 'large' may also be available."
        ),
    )
    parser.add_argument("--limit", type=int, default=60_000)
    parser.add_argument(
        "--max-raw-words",
        type=int,
        default=None,
        help=(
            "Optional cap on raw wordfreq words inspected before filtering. "
            "Default: inspect everything wordfreq exposes for the selected list."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("spanish_whitelist.txt"))
    parser.add_argument(
        "--ranked-output",
        type=Path,
        default=Path("spanish_whitelist_ranked.tsv"),
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=Path("spanish_whitelist_rejected.tsv"),
    )
    parser.add_argument("--stoplist", type=Path, default=None)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument(
        "--reject-acronyms",
        action="store_true",
        help="Reject raw all-caps tokens such as ONU, PSOE, FBI, EEUU.",
    )
    parser.add_argument(
        "--reject-roman-numerals",
        action="store_true",
        help=(
            "Reject strict Roman numerals such as ii, iv, xvi. "
            "Default is to keep them to avoid false rejects like mi/mil/civil."
        ),
    )
    parser.add_argument(
        "--keep-internal-camelcase",
        action="store_true",
        help="Keep raw mixed-case glued tokens if encountered.",
    )

    args = parser.parse_args()

    stoplist = load_stoplist(args.stoplist)

    # normalized token -> best observed score
    score_by_token: dict[str, float] = {}

    # normalized token -> raw examples
    raw_examples_by_token: dict[str, set[str]] = {}

    rejected: list[tuple[str, str, float, str]] = []

    raw_seen = 0

    for raw in iter_wordfreq_words(args.language, args.wordlist, args.max_raw_words):
        raw_seen += 1
        normalized = normalize_spanish_token(raw)

        # Get score for audit/ranking. iter_wordlist is already frequency-ordered,
        # but normalized dedupe needs a stable score for merged variants.
        score = zipf_score(raw, args.language, args.wordlist)

        keep, reason = should_keep(
            raw,
            normalized,
            min_len=args.min_len,
            max_len=args.max_len,
            reject_acronyms=args.reject_acronyms,
            reject_roman_numerals=args.reject_roman_numerals,
            keep_internal_camelcase=args.keep_internal_camelcase,
            stoplist=stoplist,
        )

        if keep:
            old_score = score_by_token.get(normalized)
            if old_score is None or score > old_score:
                score_by_token[normalized] = score
            raw_examples_by_token.setdefault(normalized, set()).add(raw)
        else:
            rejected.append((raw, normalized, score, reason))

    ranked = sorted(score_by_token.items(), key=lambda kv: (-kv[1], kv[0]))

    top = ranked[: args.limit]
    top_tokens = {token for token, _score in top}
    final_tokens = sorted(top_tokens)

    args.output.write_text("\n".join(final_tokens) + "\n", encoding="utf-8")

    with args.ranked_output.open("w", encoding="utf-8", newline="") as f:
        f.write("rank\tword\tzipf_frequency\traw_examples\n")
        for rank, (token, score) in enumerate(ranked, start=1):
            examples = sorted(raw_examples_by_token.get(token, []))
            example_text = ", ".join(examples[:10])
            if len(examples) > 10:
                example_text += f", ...(+{len(examples) - 10})"
            f.write(f"{rank}\t{token}\t{score:.6f}\t{example_text}\n")

    with args.reject_output.open("w", encoding="utf-8", newline="") as f:
        f.write("raw_word\tnormalized\tzipf_frequency\treason\n")
        for raw, normalized, score, reason in sorted(
            rejected,
            key=lambda row: (-row[2], row[3], row[1]),
        ):
            f.write(f"{raw}\t{normalized}\t{score:.6f}\t{reason}\n")

    print(f"Raw wordfreq words inspected: {raw_seen}", file=sys.stderr)
    print(f"Kept normalized candidates:   {len(score_by_token)}", file=sys.stderr)
    print(f"Final whitelist words:        {len(final_tokens)}", file=sys.stderr)
    print(f"Wrote whitelist:              {args.output}", file=sys.stderr)
    print(f"Wrote ranked-token audit:     {args.ranked_output}", file=sys.stderr)
    print(f"Wrote rejected-token audit:   {args.reject_output}", file=sys.stderr)
    print(f"Language:                     {args.language}", file=sys.stderr)
    print(f"wordfreq wordlist:            {args.wordlist}", file=sys.stderr)

    if args.reject_acronyms:
        print("Acronym policy:               rejected raw all-caps tokens", file=sys.stderr)
    else:
        print("Acronym policy:               kept raw all-caps tokens if otherwise valid", file=sys.stderr)

    if args.keep_internal_camelcase:
        print("Internal CamelCase policy:    kept raw mixed-case glued tokens", file=sys.stderr)
    else:
        print("Internal CamelCase policy:    rejected raw mixed-case glued tokens", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
