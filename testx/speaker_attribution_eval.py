"""
Evaluate the LLM speaker attribution prototype (testx/speaker_attribution.py)
against the Project Dialogism Novel Corpus (PDNC): 28 novels with every
quotation hand-annotated with its speaker.
https://github.com/Priya22/project-dialogism-novel-corpus (CC-BY 4.0)

Workflow, per novel (names as in the corpus's data/ folder; see 'novels'):

    ./venv-base/bin/python -m testx.speaker_attribution_eval novels
    ./venv-base/bin/python -m testx.speaker_attribution_eval prepare TheSunAlsoRises
    ./venv-base/bin/python -m testx.speaker_attribution roster /d/p/pdnc_eval/TheSunAlsoRises
    ./venv-base/bin/python -m testx.speaker_attribution assign /d/p/pdnc_eval/TheSunAlsoRises --all
    ./venv-base/bin/python -m testx.speaker_attribution_eval score TheSunAlsoRises

'prepare' segments the novel text the way the app's plain-text import does
(new-project defaults, dialog segmentation on) into <out>/project_text.json;
<out> defaults to pdnc_eval/<novel> next to the corpus folder. PDNC texts are
hard-wrapped (a line break every ~70 chars), which the app treats as paragraph
breaks, so quotes spanning wrapped lines would never pair; 'prepare' therefore
first replaces single line breaks with spaces (--keep-line-breaks disables
this). PDNC's quotation character offsets are mapped onto our text by aligning
the two on their non-whitespace characters, which must be identical
(segmentation can also drop stray whitespace, eg before a line break).

Scoring ('score'):
- Matching: each of our attributed utterances is matched to the PDNC quotation
  it overlaps most. (PDNC counts an interrupted quotation, "Hi," she said,
  "there.", as one quotation with sub-quotations; we have an utterance per
  piece.) Unmatched utterances/quotations on either side are reported as
  dialog-detection differences, not attribution errors.
- Speakers: PDNC speaker names are resolved to characters via their main
  names and aliases.
- Mapping: our roster ids aren't PDNC names, so each roster id is mapped to
  the PDNC character it most often agrees with (majority mapping; slightly
  optimistic). The mapping table shows whether that character's PDNC names
  also match the roster entry's name/aliases, as a sanity check.
- Correct: the answer's mapped character is the gold speaker; or the answer is
  OTHER and the gold speaker has no roster id mapped to them (the line gets
  the REST voice either way). UNKNOWN and NOT_SPEECH count as wrong.
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import sys
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass

from testx.speaker_attribution import (
    NOT_SPEECH,
    OTHER,
    PROJECT_TEXT_FILE_NAME,
    UNKNOWN,
    find_utterances,
    load_book,
    load_cache,
    range_text_hash,
)
from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.book_serialization import save_book_to_project_text_file
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_DEFAULT
from tts_audiobook_tool.constants_config import PROJECT_DEFAULT_SEGMENTATION_STRATEGY
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


DEFAULT_CORPUS = "/d/p/project-dialogism-novel-corpus"
TEXT_FILE = "novel_text.txt"
QUOTES_FILE = "quotation_info.csv"
CHARACTERS_FILE = "character_info.csv"
NO_CHARACTER = "?"


@dataclass
class GoldQuote:
    speaker: str                    # resolved PDNC main name (or the raw name if unresolved)
    quote_type: str                 # Explicit / Anaphoric / Implicit
    spans: list[tuple[int, int]]    # character offsets of the sub-quotations


@dataclass
class GoldCharacter:
    name: str
    aliases: set[str]
    category: str                   # major / intermediate / minor


# ---------------------------------------------------------------------------
# Corpus access

def novel_dir(corpus: str, novel: str) -> str:
    path = os.path.join(corpus, "data", novel)
    if not os.path.isdir(path):
        sys.exit(f"No novel folder {path} (see 'novels' for names)")
    return path


def default_out_dir(corpus: str, novel: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(corpus)), "pdnc_eval", novel)


def load_novel_index(corpus: str) -> dict[str, dict]:
    """Folder name -> PDNC-Novel-Index.csv row."""
    with open(os.path.join(corpus, "PDNC-Novel-Index.csv"), encoding="utf-8") as file:
        return {row["Folder Name"]: row for row in csv.DictReader(file) if row["Folder Name"]}


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


def name_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def unwrap_lines(text: str) -> str:
    """Single line breaks -> spaces (same length, so offsets are unchanged);
    blank-line paragraph breaks are kept."""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def offset_map(source: str, ours: str) -> list[int] | None:
    """For each position in `source` (and its end), the corresponding position
    in `ours`, aligning the texts on their non-whitespace characters. None if
    those differ."""
    result = [0] * (len(source) + 1)
    j = 0
    for i, char in enumerate(source):
        if char.isspace():
            result[i] = j
            continue
        while j < len(ours) and ours[j].isspace():
            j += 1
        if j >= len(ours) or ours[j] != char:
            return None
        result[i] = j
        j += 1
    if any(not char.isspace() for char in ours[j:]):
        return None
    result[len(source)] = len(ours)
    return result


def load_gold(directory: str) -> tuple[list[GoldQuote], dict[str, GoldCharacter], list[str]]:
    """Quotations (speakers resolved to character main names via aliases),
    characters by main name, and the speaker names that couldn't be resolved."""
    characters: dict[str, GoldCharacter] = {}
    with open(os.path.join(directory, CHARACTERS_FILE), encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                aliases = {str(a) for a in ast.literal_eval(row["Aliases"])}
            except (ValueError, SyntaxError):
                aliases = set()
            characters[row["Main Name"]] = GoldCharacter(row["Main Name"], aliases, row["Category"])
    by_name: dict[str, str] = {}
    for character in characters.values():  # aliases first, so main names win
        for alias in character.aliases:
            by_name.setdefault(name_key(alias), character.name)
    for character in characters.values():
        by_name[name_key(character.name)] = character.name

    quotes: list[GoldQuote] = []
    unresolved: list[str] = []
    with open(os.path.join(directory, QUOTES_FILE), encoding="utf-8") as file:
        for row in csv.DictReader(file):
            speaker = by_name.get(name_key(row["speaker"]))
            if speaker is None:
                speaker = row["speaker"]
                unresolved.append(speaker)
            spans = [(int(s), int(e)) for s, e in ast.literal_eval(row["quoteByteSpans"])]
            quote_type = row["quoteType"] if row["quoteType"] not in ("", "nan") else "?"
            quotes.append(GoldQuote(speaker, quote_type, spans))
    return quotes, characters, unresolved


# ---------------------------------------------------------------------------
# Commands

def command_novels(args: argparse.Namespace) -> None:
    index = load_novel_index(args.corpus)
    print(f"{'folder':<30} {'person':>6} {'quotes':>6}  title")
    for folder, row in sorted(index.items()):
        path = os.path.join(args.corpus, "data", folder, QUOTES_FILE)
        with open(path, encoding="utf-8") as file:
            count = sum(1 for _ in csv.DictReader(file))
        person = row["Narrative Person"].replace(".0", "")
        print(f"{folder:<30} {person:>6} {count:>6}  {row['Novel Title']}")


def command_prepare(args: argparse.Namespace) -> None:
    directory = novel_dir(args.corpus, args.novel)
    out = args.out or default_out_dir(args.corpus, args.novel)
    path = os.path.join(out, PROJECT_TEXT_FILE_NAME)
    if os.path.exists(path) and not args.force:
        sys.exit(f"{path} exists (use --force to re-create it; that doesn't touch saved attributions,"
                 " which stay valid while the text is unchanged)")
    text = read_text(os.path.join(directory, TEXT_FILE))
    if not args.keep_line_breaks:
        text = unwrap_lines(text)
    groups = PhraseGrouper.text_to_groups(
        text,
        pysbd_lang="en",
        max_words=MAX_WORDS_PER_SEGMENT_DEFAULT,
        strategy=PROJECT_DEFAULT_SEGMENTATION_STRATEGY,
        dialog_segmentation=True,
    )
    if offset_map(text, "".join(group.text for group in groups)) is None:
        sys.exit("Segmented text differs from the novel text beyond whitespace; PDNC offsets wouldn't line up.")
    title = load_novel_index(args.corpus).get(args.novel, {}).get("Novel Title", args.novel)
    book = Book(
        sections=[BookSection(phrase_groups=groups)],
        title=title,
        text_source_kind="plain_text",
        audio_source_kind="generated",
        segmentation_settings=BookSegmentationSettings(
            language_code="en",
            max_words_per_segment=MAX_WORDS_PER_SEGMENT_DEFAULT,
            strategy=PROJECT_DEFAULT_SEGMENTATION_STRATEGY,
            dialog_segmentation=True,
        ),
    )
    os.makedirs(out, exist_ok=True)
    error = save_book_to_project_text_file(path, book)
    if error:
        sys.exit(error)
    utterances = len(find_utterances(book, 0, len(groups) - 1))
    print(f"{title}: {len(groups):,} lines, {utterances:,} dialog utterances -> {path}")
    print("\nNext:")
    print(f"  ./venv-base/bin/python -m testx.speaker_attribution roster {out}")
    print(f"  ./venv-base/bin/python -m testx.speaker_attribution assign {out} --all")
    print(f"  ./venv-base/bin/python -m testx.speaker_attribution_eval score {args.novel}")


def command_score(args: argparse.Namespace) -> None:
    directory = novel_dir(args.corpus, args.novel)
    out = args.out or default_out_dir(args.corpus, args.novel)
    text = read_text(os.path.join(directory, TEXT_FILE))
    quotes, characters, unresolved = load_gold(directory)
    book = load_book(out)
    cache = load_cache(out)
    lines = book.flat_phrase_groups
    to_ours = offset_map(text, "".join(line.text for line in lines))
    if to_ours is None:
        sys.exit("The project text no longer matches the novel text; re-run 'prepare --force'.")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.text))

    def char_span(group_indices: list[int]) -> tuple[int, int]:
        return offsets[group_indices[0]], offsets[group_indices[-1] + 1]

    # Gold sub-quotation spans in our text's offsets, sorted; they don't
    # overlap, so their ends ascend too.
    spans = sorted(
        (to_ours[s], to_ours[e], index) for index, quote in enumerate(quotes) for s, e in quote.spans)
    starts = [s for s, _, _ in spans]

    def best_quote(start: int, end: int) -> int | None:
        overlap: Counter[int] = Counter()
        j = bisect_left(starts, end) - 1
        while j >= 0 and spans[j][1] > start:
            s, e, index = spans[j]
            overlap[index] += min(e, end) - max(s, start)
            j -= 1
        return overlap.most_common(1)[0][0] if overlap else None

    # Our saved, still-valid attributions (newer saves win where ranges overlap)
    saved: dict[int, tuple[list[int], str, bool]] = {}  # first line -> (lines, label, uncertain)
    for entry in sorted(cache.get("attributions", {}).values(), key=lambda e: e.get("created") or ""):
        if entry.get("text_hash") != range_text_hash(book, entry["line_start"], entry["line_end"]):
            continue
        for item in entry["utterances"]:
            saved[item["group_indices"][0]] = (
                item["group_indices"], item["speaker"], bool(item.get("uncertain")))
    if not saved:
        sys.exit("No saved attributions; run 'speaker_attribution assign ... --all' first.")

    # Detection: PDNC quotations vs the dialog the app's segmentation found
    all_utterances = find_utterances(book, 0, len(lines) - 1)
    detected = {q for u in all_utterances if (q := best_quote(*char_span(u.group_indices))) is not None}
    unattributed = sum(1 for u in all_utterances if u.group_indices[0] not in saved)

    rows: list[tuple[str, GoldQuote]] = []   # (our label, gold quote) per matched utterance
    flagged: list[tuple[str, GoldQuote]] = []  # the rows the LLM flagged as uncertain
    unmatched: Counter[str] = Counter()
    for group_indices, label, uncertain in saved.values():
        index = best_quote(*char_span(group_indices))
        if index is None:
            unmatched[label] += 1
            continue
        row = (label, quotes[index])
        rows.append(row)
        if uncertain:
            flagged.append(row)

    roster = {c["id"]: c for c in cache["characters"]}
    agreement: dict[str, Counter[str]] = {}
    for label, quote in rows:
        if label in roster:
            agreement.setdefault(label, Counter())[quote.speaker] += 1
    mapping = {char_id: counts.most_common(1)[0][0] for char_id, counts in agreement.items()}
    mapped_speakers = set(mapping.values())

    def answer_kind(label: str) -> str:
        if label in roster:
            return "roster id"
        if label.upper().startswith(OTHER):
            return OTHER
        return label  # UNKNOWN, NOT_SPEECH

    def is_correct(label: str, quote: GoldQuote) -> bool:
        if label in mapping:
            return mapping[label] == quote.speaker
        if answer_kind(label) == OTHER:
            return quote.speaker not in mapped_speakers
        return False

    def accuracy_line(title: str, subset: list[tuple[str, GoldQuote]]) -> str:
        if not subset:
            return f"  {title:<22} {0:>6}"
        correct = sum(1 for label, quote in subset if is_correct(label, quote))
        return f"  {title:<22} {len(subset):>6}  {100 * correct / len(subset):6.1f}%"

    title = book.title or args.novel
    print(f"===== {title}: speaker attribution vs PDNC =====")
    if unresolved:
        print(f"(PDNC speaker names not matching any character: {len(unresolved)} quotes,"
              f" eg {', '.join(f'{n} {c}' for n, c in Counter(unresolved).most_common(3))})")

    print("\nDialog detection")
    print(f"  PDNC quotations:              {len(quotes):>6,}; overlapped by our dialog: "
          f"{len(detected):,} ({100 * len(detected) / len(quotes):.1f}%)")
    print(f"  our dialog utterances:        {len(all_utterances):>6,}; with saved attributions: "
          f"{len(saved):,}" + (f" ({unattributed:,} not attributed yet)" if unattributed else ""))
    print(f"  attributed, matching a PDNC quotation: {len(rows):,}; not matching any: "
          f"{sum(unmatched.values()):,}" + (f" ({', '.join(f'{k} {v}' for k, v in unmatched.most_common(4))})"
                                            if unmatched else ""))

    print(f"\nAttribution accuracy (matched utterances)  {'n':>6}  {'acc':>7}")
    print(accuracy_line("all", rows))
    for kind in ("roster id", OTHER, UNKNOWN, NOT_SPEECH):
        print(accuracy_line(f"answer: {kind}", [r for r in rows if answer_kind(r[0]) == kind]))
    for quote_type in sorted({q.quote_type for _, q in rows}):
        print(accuracy_line(f"type: {quote_type}", [r for r in rows if r[1].quote_type == quote_type]))
    for category in ("major", "intermediate", "minor"):
        print(accuracy_line(f"speaker: {category}", [
            r for r in rows if characters.get(r[1].speaker, GoldCharacter("", set(), NO_CHARACTER)).category == category]))
    # Whether the LLM's own "uncertain" flags point at its errors
    flagged_ids = {id(row) for row in flagged}
    print(accuracy_line("flagged uncertain", flagged))
    print(accuracy_line("not flagged", [r for r in rows if id(r) not in flagged_ids]))
    errors = [r for r in rows if not is_correct(*r)]
    if errors:
        caught = sum(1 for r in errors if id(r) in flagged_ids)
        print(f"  (flags catch {caught} of {len(errors)} errors, {100 * caught / len(errors):.0f}%)")

    print("\nRoster id -> PDNC character (majority mapping)")
    print(f"  {'id':<24} {'n':>5} {'agree':>6}  {'PDNC character':<28} names match")
    for char_id, counts in sorted(agreement.items(), key=lambda item: -sum(item[1].values())):
        total = sum(counts.values())
        speaker = mapping[char_id]
        entry = roster[char_id]
        ours = {name_key(entry["name"]), *(name_key(a) for a in entry["aliases"])}
        gold = characters.get(speaker)
        theirs = {name_key(speaker), *(name_key(a) for a in gold.aliases)} if gold else {name_key(speaker)}
        narrator = " [narrator]" if entry["is_narrator"] else ""
        print(f"  {char_id:<24} {total:>5} {100 * counts[speaker] / total:5.1f}%  {speaker[:28]:<28} "
              f"{'yes' if ours & theirs else 'NO'}{narrator}")
    unused = [c for c in roster if c not in agreement]
    if unused:
        print(f"  (roster ids with no matched utterances: {', '.join(unused)})")

    spoken = Counter(quote.speaker for quote in quotes)
    missing = [
        name for name, character in characters.items()
        if character.category in ("major", "intermediate") and name not in mapped_speakers and spoken[name]
    ]
    print(f"\nPDNC major/intermediate characters with no roster id: "
          + (", ".join(f"{name} ({characters[name].category}, {spoken[name]} quotes)" for name in missing)
             or "none"))

    confusions = Counter(
        (quote.speaker, label) for label, quote in rows if not is_correct(label, quote))
    if confusions:
        print("\nMost common errors (gold speaker -> our answer)")
        for (speaker, label), count in confusions.most_common(args.errors):
            print(f"  {count:>5}  {speaker} -> {label}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate speaker attribution against PDNC")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help=f"PDNC clone (default {DEFAULT_CORPUS})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_novels = sub.add_parser("novels", help="list the corpus novels")
    p_novels.set_defaults(func=command_novels)

    for name, func, help_text in (
            ("prepare", command_prepare, "segment a novel into a project_text.json"),
            ("score", command_score, "score saved attributions against the PDNC annotations")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("novel", help="folder name in the corpus's data/ (see 'novels')")
        p.add_argument("--out", help="project folder (default: pdnc_eval/<novel> next to the corpus)")
        p.set_defaults(func=func)
        if name == "prepare":
            p.add_argument("--force", action="store_true", help="re-create an existing project_text.json")
            p.add_argument("--keep-line-breaks", action="store_true",
                           help="don't unwrap the hard-wrapped text (single line breaks -> spaces)")
        else:
            p.add_argument("--errors", type=int, default=10, help="how many common errors to list")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
