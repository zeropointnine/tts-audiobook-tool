"""
Prototype: LLM-based speaker attribution for dialog lines.

Builds on the app's existing dialog segmentation (a project imported with
"dialog segmentation" enabled has every dialog phrase group preassigned
voice_index 1). Nothing here re-runs the dialog detector; the stored
project text is the source of truth.

Steps, all cached in <project>/speaker_attribution.json:

    # 1) roster: full-book pass that builds the speaking-character list (the
    #    canonical speaker ids). Re-running it discards all saved attributions.
    ./venv-base/bin/python -m testx.speaker_attribution roster /d/p/leech

    # list: sections with their line ranges, attribution coverage, the roster
    ./venv-base/bin/python -m testx.speaker_attribution list /d/p/leech

    # counts: speaker id counts from saved attributions, descending; --lines
    # takes a 1-based inclusive range (unlike --line-start/--line-end)
    ./venv-base/bin/python -m testx.speaker_attribution counts /d/p/leech
    ./venv-base/bin/python -m testx.speaker_attribution counts /d/p/leech --lines 252-481

    # 2) assign: attribute every dialog utterance to a roster id (or OTHER,
    #    UNKNOWN, NOT_SPEECH) and save the results; the project isn't changed.
    #    --all processes the whole book in chunks (see plan_chunks), in
    #    reading order, one fresh LLM session per chunk; the context before
    #    each chunk carries already-attributed speakers. Resumable: re-running
    #    skips chunks with valid saved results (and retries failed ones).
    ./venv-base/bin/python -m testx.speaker_attribution assign /d/p/leech --all --plan
    ./venv-base/bin/python -m testx.speaker_attribution assign /d/p/leech --all
    ./venv-base/bin/python -m testx.speaker_attribution assign /d/p/leech --section 4
    ./venv-base/bin/python -m testx.speaker_attribution assign /d/p/leech --line-start 300 --line-end 420

    # 3) apply: map speakers to voices and write voice indices into
    #    project_text.json (backup first), from saved results; no LLM call.
    ./venv-base/bin/python -m testx.speaker_attribution apply /d/p/leech --all \
        --characters "NARRATOR=3, BARON=4, DIDIER=5, HELENE=6, REST=2"
    ./venv-base/bin/python -m testx.speaker_attribution apply /d/p/leech --section 4 --characters "BARON=4, REST=2"

Ranges (assign / apply):
    --section N             1-based section number (as shown by 'list')
    --line-start A --line-end B
                            book-wide line range, 0-based and inclusive; a
                            "line" is a phrase group (a text segment). It may
                            cover part of a section or span several.
    --all                   assign: the whole book in chunks
                            (--chunk-words N, default 5000; --plan to preview);
                            apply: every saved range (newer results win)

Useful flags:
    --dump-prompt FILE      write the prompt to FILE and exit (no LLM call)
    --from-response FILE    use a saved LLM reply instead of calling the LLM
    --characters "ID=N, ..., REST=N"
                            apply (required): roster ids mapped to 1-based UI
                            voice numbers; REST = everything else spoken
                            (unlisted characters, OTHER, UNKNOWN), default 2
    --show-llm-fields       roster/list: also print aliases and descriptions.
                            Roster ids, names and narration notes are kept
                            spoiler-free; aliases and descriptions are
                            LLM-facing and complete, so they may contain spoilers.

LLM endpoint/model/key/extra params come from the installed app's prefs
(the same values the app's voice chat uses).

Definitions:
    dialog group  - a phrase group with voice_index >= 1
    utterance     - a dialog group that starts with an opening quote mark,
                    plus any directly following dialog groups that don't
                    (continuation pieces of the same quotation)

Voice mapping ('apply --characters'; UI "voice N" = voice_index N - 1):
    listed ids      the given voice; the narrator's dialog only gets its own
                    voice if NARRATOR (or the narrator's id) is listed
    REST            everything else spoken: unlisted characters, OTHER, UNKNOWN
                    (default voice 2, the app's dialog-detection voice)
    NOT_SPEECH      voice 1, with narration
Narration (voice_index -1 / 0) is never touched. Because narration is never
>= 1 and every dialog voice normally is, a range can be re-attributed after
applying (not so for dialog mapped to voice 1). Hand-edited dialog voice
assignments are overwritten.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import tts_audiobook_tool.app_support  # noqa: F401  # pyright: ignore[reportUnusedImport]  (import order: avoids prefs circular import)
from tts_audiobook_tool.l import L
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.book_serialization import (
    BOOK_FORMAT,
    load_book_from_project_text_file,
)


PROJECT_TEXT_FILE_NAME = "project_text.json"
CACHE_FILE_NAME = "speaker_attribution.json"
CACHE_VERSION = 1

OPENING_QUOTES = ('"', "“", "„")

NOT_SPEECH = "NOT_SPEECH"
OTHER = "OTHER"
UNKNOWN = "UNKNOWN"
RESERVED_IDS = {NOT_SPEECH, OTHER, UNKNOWN}

# 'apply' voice mapping: NOT_SPEECH lines go to the narration voice; REST (in
# --characters) is everything spoken that isn't listed, defaulting to voice 2
# (the voice the app's dialog detection preassigns).
NOT_SPEECH_VOICE_INDEX = 0
REST_KEYWORD = "REST"
DEFAULT_REST_VOICE = 2  # 1-based UI voice number

# Approximate size of the context text given before each attributed range
PREVIOUS_CONTEXT_CHARS = 2500

# 'assign --all' chunking (see plan_chunks)
DEFAULT_CHUNK_WORDS = 5000
MIN_PIECE_WORDS = 20  # smaller break-delimited pieces (headings etc) join the following text
CHUNK_BREAK_REASONS = (Reason.SPACE_BREAK, Reason.SECTION_BREAK)
CHUNK_RETRY_DELAYS = (10, 30)  # seconds before the 2nd and 3rd attempt at a failed chunk


# ---------------------------------------------------------------------------
# Book / utterance model

@dataclass
class Utterance:
    id: int                 # 1-based, unique within the examined line range
    group_indices: list[int]  # book-wide (flat) 0-based phrase group indices
    text: str


def load_book(project_dir: str) -> Book:
    path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
    result = load_book_from_project_text_file(path)
    if isinstance(result, str):
        sys.exit(result)
    return result


# A "line" is a phrase group, numbered book-wide from 0 (Book.flat_phrase_groups
# order). Line ranges are inclusive: (start, end).

def section_starts(book: Book) -> list[int]:
    """Book-wide index of each section's first line."""
    starts: list[int] = []
    total = 0
    for section in book.sections:
        starts.append(total)
        total += len(section.phrase_groups)
    return starts


def section_bounds(book: Book, section_index: int) -> tuple[int, int]:
    start = section_starts(book)[section_index]
    return start, start + len(book.sections[section_index].phrase_groups) - 1


def section_of_line(book: Book, line: int) -> int:
    starts = section_starts(book)
    return max(i for i, start in enumerate(starts) if start <= line)


def range_text(book: Book, start: int, end: int) -> str:
    return "".join(group.text for group in book.flat_phrase_groups[start:end + 1])


def range_text_hash(book: Book, start: int, end: int) -> str:
    return hashlib.sha1(range_text(book, start, end).encode("utf-8")).hexdigest()[:16]


def section_text_hash(book: Book, section_index: int) -> str:
    return range_text_hash(book, *section_bounds(book, section_index))


def range_key(start: int, end: int) -> str:
    return f"{start}-{end}"


def find_utterances(book: Book, start: int, end: int) -> list[Utterance]:
    """Group the dialog lines within [start, end] into utterances. An
    utterance never continues across a section boundary."""
    first_lines = set(section_starts(book))
    utterances: list[Utterance] = []
    previous_was_dialog = False
    for line in range(start, end + 1):
        group = book.flat_phrase_groups[line]
        if line in first_lines:
            previous_was_dialog = False
        is_dialog = group.voice_index >= 1
        if is_dialog:
            starts_quote = group.text.lstrip().startswith(OPENING_QUOTES)
            if previous_was_dialog and not starts_quote:
                utterances[-1].group_indices.append(line)
                utterances[-1].text += group.text
            else:
                utterances.append(Utterance(len(utterances) + 1, [line], group.text))
        previous_was_dialog = is_dialog
    return utterances


def annotate_range(
        book: Book,
        start: int,
        end: int,
        open_tag: Callable[[Utterance], str | None],
        section_headers: bool,
) -> str:
    """Text of lines [start, end] with each utterance wrapped in
    open_tag(utterance) + "</q>" (eg <q>, <q id="N">, <q speaker="X">); an
    utterance whose open_tag is None is left untagged. With section_headers, a
    header line is inserted where a new section begins after the range start."""
    groups = book.flat_phrase_groups
    first_lines = set(section_starts(book))
    by_first_line = {u.group_indices[0]: u for u in find_utterances(book, start, end)}
    parts: list[str] = []
    line = start
    while line <= end:
        if section_headers and line != start and line in first_lines:
            parts.append(f"\n===== {section_label(book, section_of_line(book, line))} =====\n")
        utterance = by_first_line.get(line)
        tag = open_tag(utterance) if utterance else None
        if utterance is None or tag is None:
            parts.append(groups[line].text)
            line += 1
            continue
        # Keep trailing whitespace/newlines outside the tag.
        body = utterance.text.rstrip()
        trailing = utterance.text[len(body):]
        parts.append(f"{tag}{body}</q>{trailing}")
        line = utterance.group_indices[-1] + 1
    return "".join(parts)


def section_label(book: Book, section_index: int) -> str:
    title = book.sections[section_index].title.strip() or "(untitled)"
    return f"Section {section_index + 1}: {title}"


def range_label(book: Book, start: int, end: int) -> str:
    first, last = section_of_line(book, start), section_of_line(book, end)
    if first == last and (start, end) == section_bounds(book, first):
        return f"{section_label(book, first)} (lines {start}-{end})"
    if first == last:
        return f"Lines {start}-{end} (in {section_label(book, first)})"
    return f"Lines {start}-{end} ({section_label(book, first)} to {section_label(book, last)})"


def resolve_range(args: argparse.Namespace, book: Book) -> tuple[int, int]:
    """--section N (1-based) or --line-start/--line-end (0-based, inclusive)."""
    has_lines = args.line_start is not None or args.line_end is not None
    if args.section is not None and has_lines:
        sys.exit("Use either --section or --line-start/--line-end, not both.")
    if args.section is not None:
        if not 1 <= args.section <= len(book.sections):
            sys.exit(f"--section must be 1..{len(book.sections)}")
        return section_bounds(book, args.section - 1)
    if args.line_start is None or args.line_end is None:
        sys.exit("Give --section N, or both --line-start and --line-end.")
    last_line = len(book.flat_phrase_groups) - 1
    if not 0 <= args.line_start <= args.line_end <= last_line:
        sys.exit(f"Line range must satisfy 0 <= --line-start <= --line-end <= {last_line}")
    return args.line_start, args.line_end


def range_args_text(args: argparse.Namespace) -> str:
    """The range arguments as given, for suggesting a follow-up command."""
    if args.section is not None:
        return f"--section {args.section}"
    return f"--line-start {args.line_start} --line-end {args.line_end}"


# ---------------------------------------------------------------------------
# Cache

def cache_path(project_dir: str) -> str:
    return os.path.join(project_dir, CACHE_FILE_NAME)


def load_cache(project_dir: str) -> dict:
    path = cache_path(project_dir)
    if not os.path.exists(path):
        sys.exit(f"No roster cache at {path}. Run the 'roster' step first.")
    with open(path, "r", encoding="utf-8") as file:
        cache = json.load(file)
    if cache.get("version") != CACHE_VERSION:
        sys.exit(f"Unsupported cache version in {path}")
    return cache


def save_cache(project_dir: str, cache: dict) -> None:
    path = cache_path(project_dir)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)
    print(f"Saved {path}")


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# LLM plumbing

def make_llm(system_prompt: str) -> LlmSession:
    """
    LLM session configured exactly from the app prefs (url, key, model, extra
    params), the same way the app's voice chat builds its session; the script
    adds no model-behavior overrides of its own. Reasoning, effort etc are
    therefore controlled by the prefs' extra params.

    The one difference is the transport read timeout: a full-book request can
    take minutes, so the gap allowed between streamed chunks is longer.
    """
    prefs = Prefs.load(save_if_dirty=False)
    if not prefs.llm_url or not prefs.llm_model:
        sys.exit("LLM URL/model not set in app prefs (Options > LLM settings).")
    if not prefs.llm_api_key:
        sys.exit(f"No LLM API key: environment variable {prefs.llm_api_key_env_var!r} is empty.")
    print(f"  LLM: {prefs.llm_model} @ {prefs.llm_url}  extra params: {prefs.llm_extra_params}")
    return LlmSession(
        api_endpoint_url=prefs.llm_url,
        token=prefs.llm_api_key,
        model=prefs.llm_model,
        system_prompt=system_prompt,
        extra_params=prefs.llm_extra_params,
        timeout=(15.0, 900.0),
    )


def llm_description() -> dict:
    prefs = Prefs.load(save_if_dirty=False)
    return {"url": prefs.llm_url, "model": prefs.llm_model, "extra_params": prefs.llm_extra_params}


REASONING_DUMP_PATH = os.path.join(tempfile.gettempdir(), "speaker_attribution_reasoning.txt")


def send_streaming(llm: LlmSession, message: str) -> str:
    """Send with streaming so long calls show progress. The model's reasoning
    (if any) is saved to REASONING_DUMP_PATH for inspection."""
    received = 0

    def on_chunk(chunk: str) -> None:
        nonlocal received
        received += len(chunk)
        print(f"\r  receiving... {received:,} chars", end="", flush=True)

    reasoning_parts: list[str] = []
    reasoning = 0

    def on_reasoning(chunk: str) -> None:
        nonlocal reasoning
        reasoning_parts.append(chunk)
        reasoning += len(chunk)
        print(f"\r  thinking... {reasoning:,} chars", end="", flush=True)

    print(f"  sending {len(message):,} chars (~{len(message) // 4:,} tokens)")
    started = datetime.datetime.now()
    reply = llm.send(message, on_chunk=on_chunk, on_reasoning=on_reasoning)
    seconds = (datetime.datetime.now() - started).total_seconds()
    print(f"\n  done in {seconds:.0f}s: {received:,} answer chars, {reasoning:,} reasoning chars")
    if reasoning_parts:
        with open(REASONING_DUMP_PATH, "w", encoding="utf-8") as file:
            file.write("".join(reasoning_parts))
        print(f"  reasoning saved to {REASONING_DUMP_PATH}")
    if not reply.strip():
        hint = (
            " The model reasoned but never answered: it most likely hit its output-token limit"
            " while thinking (reasoning counts toward max_tokens; DeepSeek's default in"
            " thinking mode is 64K)."
            if reasoning else ""
        )
        raise RuntimeError(f"Empty reply from the LLM.{hint}")
    return reply


def extract_json(reply: str) -> dict:
    text = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in LLM reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM reply JSON is not an object")
    return value


def maybe_dump_prompt(args: argparse.Namespace, system_prompt: str, message: str) -> bool:
    if not args.dump_prompt:
        return False
    with open(args.dump_prompt, "w", encoding="utf-8") as file:
        file.write("=== SYSTEM ===\n" + system_prompt + "\n\n=== USER ===\n" + message)
    print(f"Wrote prompt to {args.dump_prompt} ({len(message):,} chars); no LLM call made.")
    return True


def read_saved_response(args: argparse.Namespace) -> str | None:
    if not args.from_response:
        return None
    with open(args.from_response, "r", encoding="utf-8") as file:
        return file.read()


# ---------------------------------------------------------------------------
# Step 1: roster (full-book pass)

ROSTER_SYSTEM_PROMPT = """\
You are a careful literary analyst helping cast a multi-voice audiobook.
You will receive the full text of a book. Passages that were detected as
quoted dialog are wrapped in <q>...</q> tags. Identify the characters who
speak that dialog.\
"""

ROSTER_INSTRUCTIONS = """\
Task: produce the roster of characters who speak dialog in this book.

Rules:
- Include every character who speaks more than a line or two of dialog,
  minor characters included, up to 40 characters.
- Order the list by how much dialog the character speaks, most first, based
  on your overall impression of the book.
- "prominence" is a coarse tier of how much dialog the character has:
  "major", "supporting" or "minor".
- Do not count, tally or list individual dialog passages, and keep your
  reasoning brief: an approximate ranking is all that is needed.
- Merge all names, titles, nicknames and descriptions that refer to the same
  character into one entry ("aliases"); include forms the text uses when
  attributing dialog (eg "the captain", "Mother", "my brother").
- If the book is narrated in the first person and the narrator speaks dialog,
  include the narrator as a character with "is_narrator": true. At most one
  character may be the narrator.
- Avoid spoilers in the user-facing fields: "id", "name" and the narration
  "notes" are shown to someone who may not have read the book. Base them on
  how the text refers to the character when they first appear and for most of
  the book, never on identities, names or events revealed later (secret
  identities, twists, deaths, late reveals). Give the first-person narrator
  the id "NARRATOR" unless the text names them from early on.
- "aliases" and "description" are not shown to the user; they are used to
  attribute dialog in individual chapters. Make them complete, including names
  and facts revealed later in the book, and anything that helps tell this
  character's dialog apart from others' (eg relationships, speech habits, other
  bodies or identities they speak through).
- "id" must be UPPER_SNAKE_CASE (letters, digits and underscores only; no
  spaces), unique, and must not be OTHER, UNKNOWN or NOT_SPEECH.
- Make ids clearly distinct from each other, each one recognizably naming its
  character. When characters share a role, title or name, disambiguate with a
  distinguishing word (eg GATE_GUARD and MINE_GUARD, OLD_MARIE and YOUNG_MARIE),
  not with numbers. Never give the same character two entries or two ids.

Respond with only a JSON object, no commentary, in this shape:
{
  "narration": {"person": "first" | "third" | "mixed", "notes": "<one spoiler-free sentence>"},
  "characters": [
    {
      "id": "ELIZABETH",
      "name": "Elizabeth Bennet",
      "aliases": ["Lizzy", "Miss Bennet", "her sister"],
      "gender": "male" | "female" | "unknown",
      "description": "<one or two sentences: who they are, spoilers allowed>",
      "is_narrator": false,
      "prominence": "major" | "supporting" | "minor"
    }
  ]
}\
"""


def build_roster_message(book: Book) -> str:
    sections = []
    for section_index in range(len(book.sections)):
        start, end = section_bounds(book, section_index)
        sections.append(
            f"===== {section_label(book, section_index)} =====\n"
            + annotate_range(book, start, end, lambda _: "<q>", section_headers=False).strip()
        )
    title = book.title.strip() or "(unknown title)"
    return (
        f"Book title: {title}\n\n"
        + ROSTER_INSTRUCTIONS
        + "\n\n===== BOOK TEXT BEGINS =====\n\n"
        + "\n\n".join(sections)
        + "\n\n===== BOOK TEXT ENDS =====\n\n"
        + "Now respond with the JSON roster."
    )


PROMINENCE_RANK = {"major": 3, "supporting": 2, "minor": 1}


def validate_roster(value: dict) -> tuple[dict, list[dict]]:
    """Normalize the LLM's roster. Ids: uppercase, no spaces or punctuation
    (runs of underscores collapse to one), not reserved, unique. An entry
    whose id duplicates an earlier one is merged into it when it looks like
    the same character (see same_character), otherwise kept under a suffixed
    id (GUARD_2). At most one character is the narrator."""
    raw_characters = value.get("characters")
    if not isinstance(raw_characters, list) or not raw_characters:
        raise ValueError("Roster has no 'characters' list")
    characters: list[dict] = []
    by_id: dict[str, dict] = {}
    for raw in raw_characters:
        if not isinstance(raw, dict):
            continue
        char_id = re.sub(r"[^A-Z0-9_]+", "_", str(raw.get("id", "")).upper())
        char_id = re.sub(r"_+", "_", char_id).strip("_")
        if not char_id or char_id in RESERVED_IDS:
            print(f"  (skipping roster entry with bad id: {raw.get('id')!r})")
            continue
        aliases = raw.get("aliases")
        character = {
            "id": char_id,
            "name": str(raw.get("name", char_id)),
            "aliases": [str(a) for a in aliases] if isinstance(aliases, list) else [],
            "gender": str(raw.get("gender", "unknown")),
            "description": str(raw.get("description", "")),
            "is_narrator": bool(raw.get("is_narrator")),
            "prominence": str(raw.get("prominence", "")),
        }
        existing = by_id.get(char_id)
        if existing and same_character(existing, character):
            merge_character(existing, character)
            print(f"  (merged duplicate roster entry {char_id}: {character['name']!r})")
            continue
        if existing:
            suffix = 2
            while f"{char_id}_{suffix}" in by_id:
                suffix += 1
            character["id"] = f"{char_id}_{suffix}"
            print(f"  (duplicate roster id {char_id} for a different character"
                  f" {character['name']!r}; renamed {character['id']})")
        by_id[character["id"]] = character
        characters.append(character)

    narrator_seen = False
    for character in characters:
        character["is_narrator"] = character["is_narrator"] and not narrator_seen
        narrator_seen = narrator_seen or character["is_narrator"]

    raw_narration = value.get("narration")
    narration: dict = raw_narration if isinstance(raw_narration, dict) else {}
    return narration, characters


def name_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def same_character(a: dict, b: dict) -> bool:
    """Same name, or one's name among the other's aliases. (Alias-to-alias
    overlap isn't enough: generic aliases like "madame" are often shared.)"""
    name_a, name_b = name_key(a["name"]), name_key(b["name"])
    return (name_a == name_b
            or name_a in {name_key(alias) for alias in b["aliases"]}
            or name_b in {name_key(alias) for alias in a["aliases"]})


def merge_character(target: dict, other: dict) -> None:
    """Fold a duplicate entry into the first one: union of names/aliases, the
    first non-empty description and known gender, the higher prominence."""
    known = {name_key(target["name"])} | {name_key(alias) for alias in target["aliases"]}
    for alias in [other["name"], *other["aliases"]]:
        if name_key(alias) not in known:
            target["aliases"].append(alias)
            known.add(name_key(alias))
    target["description"] = target["description"] or other["description"]
    if target["gender"] == "unknown":
        target["gender"] = other["gender"]
    if PROMINENCE_RANK.get(other["prominence"], 0) > PROMINENCE_RANK.get(target["prominence"], 0):
        target["prominence"] = other["prominence"]
    target["is_narrator"] = target["is_narrator"] or other["is_narrator"]


def command_roster(args: argparse.Namespace) -> None:
    book = load_book(args.project_dir)
    if not book.segmentation_settings.dialog_segmentation:
        print("Warning: this book was not segmented with dialog segmentation enabled;"
              " dialog lines may not be marked.")
    total_utterances = sum(
        len(find_utterances(book, *section_bounds(book, i))) for i in range(len(book.sections)))
    print(f"{book.title or '(untitled)'}: {len(book.sections)} sections, {total_utterances:,} utterances")
    if total_utterances == 0:
        sys.exit("No dialog lines (voice_index >= 1) found.")

    message = build_roster_message(book)
    if maybe_dump_prompt(args, ROSTER_SYSTEM_PROMPT, message):
        return

    reply = read_saved_response(args)
    if reply is None:
        llm = make_llm(ROSTER_SYSTEM_PROMPT)
        reply = send_streaming(llm, message)
    narration, characters = validate_roster(extract_json(reply))

    cache = {
        "version": CACHE_VERSION,
        "created": now_iso(),
        "llm": llm_description(),
        "book_title": book.title,
        "section_text_hashes": [section_text_hash(book, i) for i in range(len(book.sections))],
        "narration": narration,
        "characters": characters,
        "roster_raw_reply": reply,
        "attributions": {},
    }
    save_cache(args.project_dir, cache)
    print_roster(cache, args.show_llm_fields)


def print_roster(cache: dict, show_llm_fields: bool) -> None:
    """Print the roster's user-facing (spoiler-free) fields; descriptions and
    aliases are LLM-facing and may contain spoilers, so they're shown only
    when show_llm_fields is set."""
    narration = cache.get("narration", {})
    if narration:
        print(f"\nNarration: {narration.get('person', '?')} person. {narration.get('notes', '')}")
    print(f"\n{'#':>3}  {'id':<20} {'tier':<10} {'gender':<7} name")
    for i, character in enumerate(cache["characters"], 1):
        narrator = " [NARRATOR]" if character["is_narrator"] else ""
        tier = str(character.get("prominence") or "?")
        print(f"{i:>3}  {character['id']:<20} {tier:<10} {character['gender']:<7} "
              f"{character['name']}{narrator}")
        if show_llm_fields:
            indent = " " * 43
            if character["aliases"]:
                print(f"{indent} aliases: {', '.join(character['aliases'])}")
            if character["description"]:
                print(f"{indent} {character['description']}")


# ---------------------------------------------------------------------------
# list

def command_list(args: argparse.Namespace) -> None:
    book = load_book(args.project_dir)
    cache = None
    if os.path.exists(cache_path(args.project_dir)):
        cache = load_cache(args.project_dir)
    saved_ranges = [
        (item["line_start"], item["line_end"], len(item["utterances"]))
        for item in (cache or {}).get("attributions", {}).values()
    ]
    # assigned: "yes" = saved results for exactly this section, "part" = a
    # saved line range overlaps it
    print(f"{'sect':>4}  {'lines':<11}  {'utter.':>6}  {'assigned':<8} title")
    for i, section in enumerate(book.sections):
        start, end = section_bounds(book, i)
        assigned = ""
        if any((s, e) == (start, end) for s, e, _ in saved_ranges):
            assigned = "yes"
        elif any(s <= end and e >= start for s, e, _ in saved_ranges):
            assigned = "part"
        print(f"{i + 1:>4}  {f'{start}-{end}':<11}  "
              f"{len(find_utterances(book, start, end)):>6}  {assigned:<8} {section.title.strip()}")
    if saved_ranges:
        print("\nSaved line ranges: " + ", ".join(
            f"{s}-{e} ({n} utterances)" for s, e, n in sorted(saved_ranges)))
    if cache:
        print_roster(cache, args.show_llm_fields)


def command_counts(args: argparse.Namespace) -> None:
    """Speaker id counts from saved attributions, for --lines A-B (1-based,
    inclusive) or the whole book."""
    book = load_book(args.project_dir)
    cache = load_cache(args.project_dir)
    line_count = len(book.flat_phrase_groups)
    if args.lines:
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", args.lines)
        if not match:
            sys.exit("--lines must look like A-B, eg 252-481 (1-based, inclusive)")
        first, last = int(match[1]), int(match[2])
        if not 1 <= first <= last <= line_count:
            sys.exit(f"--lines must satisfy 1 <= A <= B <= {line_count}")
        start, end = first - 1, last - 1
        title = f"Speaker counts, lines {first}-{last}"
    else:
        start, end = 0, line_count - 1
        title = "Speaker counts, all lines"
    counts, unattributed = speaker_counts(cache, book, start, end)
    print_speaker_counts(title, counts, unattributed)


# ---------------------------------------------------------------------------
# Step 2: assignment (attribute each dialog utterance to a roster character)

ASSIGN_SYSTEM_PROMPT = """\
You are a careful literary analyst performing quotation attribution for a
multi-voice audiobook: for each quoted dialog passage you decide which
character speaks it.\
"""

ASSIGN_INSTRUCTIONS = """\
Below is a passage of a book (a chapter, part of one, or several). Each
detected dialog passage is wrapped in <q id="N">...</q>. For EVERY id, decide
who speaks it.

The passage may be preceded by context text. In the context, dialog that was
already attributed is wrapped in <q speaker="...">...</q>; use it to follow the
conversation, but give answers only for the ids in the passage.

Allowed answers:
- a character id from the roster
- "OTHER:<short name>" for a speaker not in the roster (eg "OTHER:guard")
- "UNKNOWN" if the speaker cannot be determined
- "NOT_SPEECH" if the tagged text is not spoken dialog at all (eg a quoted
  title, sign, label, scare quote or written text)

Guidance:
- Follow explicit attributions ("said X", "X asks") and pronoun attributions
  by resolving who the pronoun refers to.
- For untagged lines in a back-and-forth exchange, track turn-taking.
- Consecutive tags in the same paragraph separated only by an attribution
  (eg <q>"Hi,"</q> she said, <q>"there."</q>) almost always share a speaker.
- In first-person narration, dialog attributed to "I" belongs to the narrator character.

Respond with only a JSON object, no commentary:
{"speakers": {"1": "<answer>", "2": "<answer>", ...}, "uncertain": [<ids you are unsure about>]}\
"""


def roster_for_prompt(characters: list[dict]) -> str:
    lines = []
    for character in characters:
        narrator = " (the first-person narrator)" if character["is_narrator"] else ""
        aliases = ", ".join(character["aliases"])
        lines.append(
            f"- {character['id']}: {character['name']}{narrator}; {character['gender']}; "
            f"aliases: {aliases or '-'}; {character['description']}"
        )
    return "\n".join(lines)


def saved_speakers_by_line(cache: dict, book: Book) -> dict[int, str]:
    """First line of each saved, still-valid utterance -> its speaker label
    (later saves win where saved ranges overlap)."""
    speakers: dict[int, str] = {}
    for entry in sorted(cache.get("attributions", {}).values(), key=lambda e: e.get("created") or ""):
        if entry.get("text_hash") != range_text_hash(book, entry["line_start"], entry["line_end"]):
            continue
        for item in entry["utterances"]:
            speakers[item["group_indices"][0]] = item["speaker"]
    return speakers


def context_before(book: Book, start: int, speakers_by_line: dict[int, str]) -> str:
    """Whole lines directly before `start` (about PREVIOUS_CONTEXT_CHARS of
    text), with already-attributed utterances tagged by their speaker. This
    carries the conversation state across range boundaries."""
    first = start
    size = 0
    while first > 0 and size < PREVIOUS_CONTEXT_CHARS:
        first -= 1
        size += len(book.flat_phrase_groups[first].text)
    if first == start:
        return ""

    def open_tag(utterance: Utterance) -> str | None:
        speaker = speakers_by_line.get(utterance.group_indices[0])
        return f'<q speaker="{speaker}">' if speaker else None

    return annotate_range(book, first, start - 1, open_tag, section_headers=True).strip()


def build_assign_message(
        book: Book, start: int, end: int, cache: dict, speakers_by_line: dict[int, str],
) -> str:
    parts = [ASSIGN_INSTRUCTIONS, "", "Character roster:", roster_for_prompt(cache["characters"])]
    narration = cache.get("narration", {})
    if narration.get("person"):
        parts.append(f"\nThe book is narrated in the {narration['person']} person.")
    context = context_before(book, start, speakers_by_line)
    if context:
        parts += ["", "===== CONTEXT: PRECEDING TEXT (do not attribute) =====", context]
    parts += [
        "",
        f"===== TEXT TO ATTRIBUTE: {range_label(book, start, end)} =====",
        annotate_range(book, start, end, lambda u: f'<q id="{u.id}">', section_headers=True).strip(),
        "===== TEXT ENDS =====",
        "",
        "Now respond with the JSON object covering every id.",
    ]
    return "\n".join(parts)


def normalize_answer(raw: object, roster_ids: set[str]) -> tuple[str, str]:
    """-> (kind, label): kind is a roster id, OTHER, UNKNOWN or NOT_SPEECH."""
    text = str(raw).strip()
    upper = text.upper()
    if upper in roster_ids:
        return upper, upper
    if upper == NOT_SPEECH or upper.replace(" ", "_") == NOT_SPEECH:
        return NOT_SPEECH, NOT_SPEECH
    if upper == UNKNOWN:
        return UNKNOWN, UNKNOWN
    if upper.startswith(OTHER):
        return OTHER, text
    # Not a roster id: keep it visible for review, bucket as OTHER.
    return OTHER, f"OTHER:{text}"


def parse_assignments(
        value: dict, utterances: list[Utterance], roster_ids: set[str],
) -> tuple[dict[int, tuple[str, str]], set[int], list[int]]:
    raw_speakers = value.get("speakers")
    if not isinstance(raw_speakers, dict):
        raise ValueError("Reply has no 'speakers' object")
    valid_ids = {u.id for u in utterances}
    result: dict[int, tuple[str, str]] = {}
    for key, raw in raw_speakers.items():
        try:
            utterance_id = int(str(key).strip())
        except ValueError:
            continue
        if utterance_id in valid_ids:
            result[utterance_id] = normalize_answer(raw, roster_ids)
    uncertain: set[int] = set()
    raw_uncertain = value.get("uncertain")
    if isinstance(raw_uncertain, list):
        for item in raw_uncertain:
            try:
                uncertain.add(int(str(item).strip()))
            except ValueError:
                pass
    missing = sorted(valid_ids - result.keys())
    return result, uncertain, missing


def attribute(
        message: str, utterances: list[Utterance], roster_ids: set[str], reply: str | None = None,
) -> tuple[dict[int, tuple[str, str]], set[int]]:
    """A speaker for every utterance: one fresh LLM session (or a saved
    reply), plus one follow-up turn for ids missing from the reply. Ids still
    missing become UNKNOWN."""
    llm = None
    if reply is None:
        llm = make_llm(ASSIGN_SYSTEM_PROMPT)
        reply = send_streaming(llm, message)
    assignments, uncertain, missing = parse_assignments(extract_json(reply), utterances, roster_ids)
    if missing and llm is not None:
        print(f"  {len(missing)} id(s) missing from reply; asking again for those")
        follow_up = (
            f"Your reply omitted these ids: {', '.join(map(str, missing))}. "
            'Respond with only a JSON object {"speakers": {...}, "uncertain": [...]} for those ids.'
        )
        retry_value = extract_json(send_streaming(llm, follow_up))
        more, more_uncertain, missing = parse_assignments(
            retry_value, [u for u in utterances if u.id in set(missing)], roster_ids)
        assignments.update(more)
        uncertain |= more_uncertain
    for utterance_id in missing:
        assignments[utterance_id] = (UNKNOWN, UNKNOWN)
    if missing:
        print(f"  {len(missing)} id(s) still missing; marked UNKNOWN")
    return assignments, uncertain


def save_attribution(
        cache: dict,
        book: Book,
        start: int,
        end: int,
        utterances: list[Utterance],
        assignments: dict[int, tuple[str, str]],
        uncertain: set[int],
) -> None:
    cache.setdefault("attributions", {})[range_key(start, end)] = {
        "created": now_iso(),
        "llm": llm_description(),
        "line_start": start,
        "line_end": end,
        "text_hash": range_text_hash(book, start, end),
        "utterances": [
            {
                "id": u.id,
                "group_indices": u.group_indices,
                "speaker": assignments[u.id][1],
                "uncertain": u.id in uncertain,
                "text": u.text.strip()[:120],
            }
            for u in utterances
        ],
    }


def warn_if_roster_stale(cache: dict, book: Book, section_indices: range) -> None:
    hashes = cache.get("section_text_hashes", [])
    if len(hashes) != len(book.sections) or any(
            hashes[i] != section_text_hash(book, i) for i in section_indices):
        print("Warning: project text changed since the roster pass "
              "(re-run 'roster' if the change is substantial).")


def command_assign(args: argparse.Namespace) -> None:
    book = load_book(args.project_dir)
    if args.all and args.plan:  # the chunk plan doesn't need the roster
        chunks = plan_chunks(book, args.chunk_words)
        print(f"{book.title or '(untitled)'}: {len(chunks)} chunks of up to {args.chunk_words:,} words")
        print_plan(book, chunks)
        return
    cache = load_cache(args.project_dir)
    if args.all:
        warn_if_roster_stale(cache, book, range(len(book.sections)))
        assign_all(args, book, cache)
        return

    start, end = resolve_range(args, book)
    warn_if_roster_stale(cache, book, range(section_of_line(book, start), section_of_line(book, end) + 1))
    utterances = find_utterances(book, start, end)
    print(f"{range_label(book, start, end)}: {len(utterances)} utterances")
    if not utterances:
        return

    message = build_assign_message(book, start, end, cache, saved_speakers_by_line(cache, book))
    if maybe_dump_prompt(args, ASSIGN_SYSTEM_PROMPT, message):
        return
    roster_ids = {c["id"] for c in cache["characters"]}
    assignments, uncertain = attribute(message, utterances, roster_ids, read_saved_response(args))
    print_review(utterances, assignments, uncertain)

    save_attribution(cache, book, start, end, utterances, assignments, uncertain)
    save_cache(args.project_dir, cache)
    print(f'\nApply with: apply {args.project_dir} {range_args_text(args)} --characters "ID=N, ..., REST=N"')


def assign_all(args: argparse.Namespace, book: Book, cache: dict) -> None:
    """Attribute the whole book, chunk by chunk in reading order, one fresh
    LLM session per chunk. Results are saved after every chunk; chunks that
    already have valid saved results are skipped, so an interrupted or
    partly failed run continues where it left off when re-run."""
    if args.dump_prompt or args.from_response:
        sys.exit("--dump-prompt/--from-response work with --section or a line range, not --all")
    chunks = plan_chunks(book, args.chunk_words)
    print(f"{book.title or '(untitled)'}: {len(chunks)} chunks of up to {args.chunk_words:,} words")
    if args.plan:
        print_plan(book, chunks)
        return

    roster_ids = {c["id"] for c in cache["characters"]}
    failures: list[str] = []
    for number, (start, end) in enumerate(chunks, 1):
        utterances = find_utterances(book, start, end)
        heading = f"\n[{number}/{len(chunks)}] {range_label(book, start, end)}: {len(utterances)} utterances"
        if not utterances:
            print(heading + " (no dialog; skipped)")
            continue
        saved = cache.get("attributions", {}).get(range_key(start, end))
        if saved and saved.get("text_hash") == range_text_hash(book, start, end):
            print(heading + " (already done)")
            continue
        print(heading)
        message = build_assign_message(book, start, end, cache, saved_speakers_by_line(cache, book))
        result = attribute_with_retries(message, utterances, roster_ids)
        if isinstance(result, str):  # keep going; failed chunks are retried by the next run
            failures.append(f"lines {start}-{end}: {result}")
            print(f"  FAILED: {result}")
            continue
        assignments, uncertain = result
        save_attribution(cache, book, start, end, utterances, assignments, uncertain)
        save_cache(args.project_dir, cache)
        print("  " + counts_text(Counter(speaker_id(label) for _, label in assignments.values())))
    print_book_summary(cache, book, failures)


def attribute_with_retries(
        message: str, utterances: list[Utterance], roster_ids: set[str],
) -> tuple[dict[int, tuple[str, str]], set[int]] | str:
    """attribute(), retried with a fresh LLM session on any failure (eg a
    dropped response stream or an unparseable reply), waiting between attempts.
    Returns the error text of the last attempt if all fail."""
    for attempt, delay in enumerate((*CHUNK_RETRY_DELAYS, None), 1):
        try:
            return attribute(message, utterances, roster_ids)
        except Exception as error:
            if delay is None:
                return f"{type(error).__name__}: {error} (after {attempt} attempts)"
            print(f"\n  attempt {attempt} failed ({type(error).__name__}: {error}); retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def counts_text(counts: Counter[str]) -> str:
    return ", ".join(f"{label} {count}" for label, count in counts.most_common())


def speaker_id(label: str) -> str:
    """A saved speaker label as an id for counting: OTHER:<name> counts as OTHER."""
    return OTHER if label.upper().startswith(OTHER) else label


def speaker_counts(cache: dict, book: Book, start: int, end: int) -> tuple[Counter[str], int]:
    """Speaker id counts over saved, still-valid utterances whose first line is
    within [start, end] (0-based), and the number of dialog utterances there
    that have no saved result."""
    speakers = saved_speakers_by_line(cache, book)
    counts = Counter(speaker_id(label) for line, label in speakers.items() if start <= line <= end)
    unattributed = sum(1 for u in find_utterances(book, start, end) if u.group_indices[0] not in speakers)
    return counts, unattributed


def print_speaker_counts(title: str, counts: Counter[str], unattributed: int = 0) -> None:
    """Speaker ids by utterance count, descending."""
    pending = f", {unattributed:,} not attributed yet" if unattributed else ""
    print(f"\n{title}: {sum(counts.values()):,} attributed utterances{pending}")
    for speaker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count:>6,}  {speaker}")


def print_book_summary(cache: dict, book: Book, failures: list[str]) -> None:
    print("\n===== Summary =====")
    counts, unattributed = speaker_counts(cache, book, 0, len(book.flat_phrase_groups) - 1)
    print_speaker_counts("Speaker counts, all lines", counts, unattributed)
    labels = list(saved_speakers_by_line(cache, book).values())
    uncertain = sum(
        1 for entry in cache.get("attributions", {}).values()
        for item in entry["utterances"] if item.get("uncertain"))
    others = Counter(
        label.split(":", 1)[1].strip().lower() or "?"
        for label in labels if label.upper().startswith(OTHER) and ":" in label
    )
    recurring = Counter({name: count for name, count in others.items() if count >= 3})
    print(f"\nFlagged uncertain: {uncertain}")
    if recurring:
        print(f"Recurring OTHER speakers (roster candidates): {counts_text(recurring)}")
    if failures:
        print(f"\nFailed chunks ({len(failures)}); re-run the same command to retry them:")
        for failure in failures:
            print(f"  {failure}")


# ---------------------------------------------------------------------------
# Chunking for 'assign --all'

def plan_chunks(book: Book, chunk_words: int) -> list[tuple[int, int]]:
    """Split the book into line ranges of up to about chunk_words words:
    1) cut after every space/section break and at section boundaries; pieces
       under MIN_PIECE_WORDS (chapter headings, dedications) join the
       following piece, so a chunk never ends on a heading;
    2) split pieces over chunk_words in two, repeatedly (see choose_split_line);
    3) merge consecutive pieces while the total stays within chunk_words."""
    groups = book.flat_phrase_groups
    words = [len(group.text.split()) for group in groups]
    first_lines = set(section_starts(book))
    last_line = len(groups) - 1

    pieces: list[tuple[int, int]] = []
    piece_start = 0
    for line, group in enumerate(groups):
        if (line == last_line or line + 1 in first_lines
                or group.phrases[-1].reason in CHUNK_BREAK_REASONS):
            pieces.append((piece_start, line))
            piece_start = line + 1

    joined: list[tuple[int, int]] = []
    pending_start: int | None = None
    for start, end in pieces:
        if pending_start is not None:
            start, pending_start = pending_start, None
        if sum(words[start:end + 1]) < MIN_PIECE_WORDS and end != last_line:
            pending_start = start
            continue
        joined.append((start, end))

    split: list[tuple[int, int]] = []
    for start, end in joined:
        split.extend(split_piece(book, words, start, end, chunk_words))

    chunks: list[tuple[int, int]] = []
    for start, end in split:
        if chunks and sum(words[chunks[-1][0]:end + 1]) <= chunk_words:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


def split_piece(book: Book, words: list[int], start: int, end: int, chunk_words: int) -> list[tuple[int, int]]:
    if start == end or sum(words[start:end + 1]) <= chunk_words:
        return [(start, end)]
    cut = choose_split_line(book, words, start, end)
    return (split_piece(book, words, start, cut, chunk_words)
            + split_piece(book, words, cut + 1, end, chunk_words))


def choose_split_line(book: Book, words: list[int], start: int, end: int) -> int:
    """The line after which to split [start, end] in two, avoiding cutting a
    conversation: a paragraph end within the middle half of the piece's words
    with the most narration (non-dialog lines) around it, then nearest the
    middle. Falls back to the paragraph end, then sentence end, nearest the
    middle, then the middle line."""
    groups = book.flat_phrase_groups
    total = sum(words[start:end + 1])

    def is_dialog(line: int) -> bool:
        return groups[line].voice_index >= 1

    def ends_with(line: int, reasons: tuple[Reason, ...]) -> bool:
        return groups[line].phrases[-1].reason in reasons

    def narration_around(line: int) -> int:
        """Consecutive non-dialog lines up to 6 before and after the cut."""
        count = 0
        for other in range(line, max(start, line - 5) - 1, -1):
            if is_dialog(other):
                break
            count += 1
        for other in range(line + 1, min(end, line + 6) + 1):
            if is_dialog(other):
                break
            count += 1
        return count

    # (line, distance of the cut from the middle, as a fraction of the words)
    candidates: list[tuple[int, float]] = []
    running = 0
    for line in range(start, end):
        running += words[line]
        candidates.append((line, abs(running - total / 2) / total))

    paragraph_reasons = (Reason.PARAGRAPH, Reason.HEADING_END)
    middle = [c for c in candidates if c[1] <= 0.25 and ends_with(c[0], paragraph_reasons)]
    if middle:
        return max(middle, key=lambda c: (narration_around(c[0]), -c[1]))[0]
    for reasons in (paragraph_reasons, (Reason.SENTENCE,)):
        options = [c for c in candidates if ends_with(c[0], reasons)]
        if options:
            return min(options, key=lambda c: c[1])[0]
    return min(candidates, key=lambda c: c[1])[0]


def print_plan(book: Book, chunks: list[tuple[int, int]]) -> None:
    print(f"\n{'#':>3}  {'lines':<11} {'words':>6} {'utter.':>6}  range")
    for number, (start, end) in enumerate(chunks, 1):
        words = sum(len(group.text.split()) for group in book.flat_phrase_groups[start:end + 1])
        print(f"{number:>3}  {f'{start}-{end}':<11} {words:>6} {len(find_utterances(book, start, end)):>6}"
              f"  {range_label(book, start, end)}")


# ---------------------------------------------------------------------------
# Step 3: apply (map speakers to voices and write voice indices)

@dataclass
class VoiceMapping:
    """Speaker -> zero-based voice_index. `by_id` holds the listed roster
    characters; everything else spoken (unlisted characters, OTHER, UNKNOWN)
    gets `rest`. NOT_SPEECH always gets the narration voice."""
    by_id: dict[str, int]
    rest: int

    def index_for(self, kind: str) -> int:
        if kind == NOT_SPEECH:
            return NOT_SPEECH_VOICE_INDEX
        return self.by_id.get(kind, self.rest)


def parse_voice_mapping(text: str, cache: dict) -> VoiceMapping:
    """Parse --characters, eg "BARON=2, DIDIER=3, REST=4": roster ids (or the
    keyword REST) mapped to 1-based UI voice numbers. REST defaults to voice
    DEFAULT_REST_VOICE."""
    known = {c["id"] for c in cache["characters"]}
    by_id: dict[str, int] = {}
    rest: int | None = None
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip().upper()
        if not sep or not key or not value.strip().isdigit() or int(value) < 1:
            sys.exit(f"Bad --characters item {item!r}: expected ID=N with a voice number N >= 1")
        if key in by_id or (key == REST_KEYWORD and rest is not None):
            sys.exit(f"--characters lists {key} more than once")
        if key == REST_KEYWORD:
            rest = int(value) - 1
        elif key in known:
            by_id[key] = int(value) - 1
        else:
            sys.exit(f"Unknown character id in --characters: {key} (see 'list'; roster ids: {', '.join(sorted(known))})")
    mapping = VoiceMapping(by_id, DEFAULT_REST_VOICE - 1 if rest is None else rest)
    if NOT_SPEECH_VOICE_INDEX in (*mapping.by_id.values(), mapping.rest):
        print("Note: dialog mapped to voice 1 looks like narration afterwards, so later"
              " 'assign' runs won't see those lines as dialog.")
    return mapping


def print_voice_map(
        cache: dict, mapping: VoiceMapping, project_dir: str, voice_counts: Counter[int],
) -> None:
    """Per voice: who it gets and how many utterances."""
    names = {c["id"]: c["name"] for c in cache["characters"]}
    members: dict[int, list[str]] = {NOT_SPEECH_VOICE_INDEX: ["narration", NOT_SPEECH]}
    for char_id, index in mapping.by_id.items():
        members.setdefault(index, []).append(f"{char_id} ({names.get(char_id, '')})")
    members.setdefault(mapping.rest, []).append(f"{REST_KEYWORD} (other characters, OTHER, UNKNOWN)")
    print("\nVoice mapping (UI voice numbers):")
    for index in sorted(members):
        print(f"  voice {index + 1}: {', '.join(members[index])}  [{voice_counts[index]} utterances]")
    project_json = os.path.join(project_dir, "project.json")
    if os.path.exists(project_json):
        with open(project_json, "r", encoding="utf-8") as file:
            voice_references = json.load(file).get("voice_references")
        # Only v4+ projects store a top-level voice list; skip the check otherwise.
        voice_count = len(voice_references) if isinstance(voice_references, list) else None
        needed = max(members) + 1
        if voice_count is not None and voice_count < needed:
            print(f"  Note: project has {voice_count} voice(s); this mapping uses {needed}.")


def write_voices(
        project_dir: str,
        book: Book,
        group_voices: list[tuple[list[int], int]],
) -> None:
    """Set voice_index on the given (book-wide) lines, in place (backup
    first). Edits the raw JSON rather than re-serializing the Book: a Book
    round trip can re-derive unrelated fields (eg phrase reasons) elsewhere
    in the file."""
    text_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
    with open(text_path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or raw.get("format") != BOOK_FORMAT:
        sys.exit(f"Writing supports only '{BOOK_FORMAT}' project text files")
    raw_groups = [group for section in raw["book"]["sections"] for group in section["phrase_groups"]]
    groups = book.flat_phrase_groups
    if len(raw_groups) != len(groups) or any(
            "".join(p["text"] for p in raw_group["phrases"]) != group.text
            for raw_group, group in zip(raw_groups, groups)):
        sys.exit("Raw project text doesn't match the loaded book; not writing.")
    changed = 0
    for group_indices, new_index in group_voices:
        for group_index in group_indices:
            if raw_groups[group_index].get("voice_index") != new_index:
                raw_groups[group_index]["voice_index"] = new_index
                changed += 1
    timestamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    backup_path = f"{text_path}.bak-{timestamp}"
    suffix = 2
    while os.path.exists(backup_path):  # never overwrite an earlier backup
        backup_path = f"{text_path}.bak-{timestamp}-{suffix}"
        suffix += 1
    shutil.copy2(text_path, backup_path)
    temp_path = text_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(raw, file, indent=4)  # same formatting as the app's book serializer
    os.replace(temp_path, text_path)
    print(f"\nUpdated {changed} phrase group(s) in {text_path}")
    print(f"Backup: {backup_path}")


def command_apply(args: argparse.Namespace) -> None:
    """Apply saved attributions to the project text, without calling the LLM:
    one section/line range (saved by an 'assign' run over exactly that range),
    or with --all every saved range, oldest first so newer results win where
    ranges overlap. The voice mapping (--characters) is chosen here."""
    book = load_book(args.project_dir)
    cache = load_cache(args.project_dir)
    mapping = parse_voice_mapping(args.characters, cache)
    attributions = cache.get("attributions", {})
    if args.all:
        entries = []
        for entry in sorted(attributions.values(), key=lambda e: e.get("created") or ""):
            start, end = entry["line_start"], entry["line_end"]
            if entry.get("text_hash") != range_text_hash(book, start, end):
                print(f"Skipping lines {start}-{end}: project text changed since it was attributed")
                continue
            entries.append(entry)
        if not entries:
            sys.exit("No saved attributions to apply; run 'assign' first.")
        print(f"Applying {len(entries)} saved range(s), "
              f"{sum(len(e['utterances']) for e in entries):,} utterances")
    else:
        start, end = resolve_range(args, book)
        saved = attributions.get(range_key(start, end))
        if not saved:
            saved_keys = ", ".join(sorted(attributions)) or "none"
            sys.exit(f"No saved attributions for lines {start}-{end}; run 'assign' over that range first."
                     f" (Saved ranges: {saved_keys})")
        if saved.get("text_hash") != range_text_hash(book, start, end):
            sys.exit("Project text of this range changed since it was attributed; re-run 'assign'.")
        if any(not start <= g <= end for item in saved["utterances"] for g in item["group_indices"]):
            sys.exit("Saved attributions don't fit the current project text; re-run 'assign'.")
        entries = [saved]
        print(f"{range_label(book, start, end)}: applying {len(saved['utterances'])} saved attributions"
              f" from {saved.get('created', '?')}")

    roster_ids = {c["id"] for c in cache["characters"]}
    group_voices: list[tuple[list[int], int]] = []
    voice_counts: Counter[int] = Counter()
    for entry in entries:
        for item in entry["utterances"]:
            voice = mapping.index_for(normalize_answer(item["speaker"], roster_ids)[0])
            group_voices.append((item["group_indices"], voice))
            voice_counts[voice] += 1
    if not args.all:
        items = entries[0]["utterances"]
        print_review(
            [Utterance(item["id"], item["group_indices"], item["text"]) for item in items],
            {item["id"]: normalize_answer(item["speaker"], roster_ids) for item in items},
            {item["id"] for item in items if item.get("uncertain")},
            mapping,
        )
    print_voice_map(cache, mapping, args.project_dir, voice_counts)
    write_voices(args.project_dir, book, group_voices)


def print_review(
        utterances: list[Utterance],
        assignments: dict[int, tuple[str, str]],
        uncertain: set[int],
        mapping: VoiceMapping | None = None,
) -> None:
    """One row per utterance; the voice column only when a voice mapping is given."""
    voice_header = f"{'voice':>5}  " if mapping is not None else ""
    print(f"\n{'id':>4}  {voice_header}{'speaker':<22} text")
    counts: Counter[str] = Counter()
    for utterance in utterances:
        kind, label = assignments[utterance.id]
        counts[speaker_id(label)] += 1
        voice = f"{mapping.index_for(kind) + 1:>5}  " if mapping is not None else ""
        flag = "?" if utterance.id in uncertain else " "
        excerpt = " ".join(utterance.text.split())
        if len(excerpt) > 70:
            excerpt = excerpt[:67] + "..."
        print(f"{utterance.id:>4}{flag} {voice}{label[:22]:<22} {excerpt}")
    print_speaker_counts("Speaker counts", counts)
    if uncertain:
        print(f"Flagged uncertain: {len(uncertain)} (marked '?')")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM speaker attribution prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("project_dir")
        p.add_argument("--dump-prompt", metavar="FILE")
        p.add_argument("--from-response", metavar="FILE")

    def add_range_args(p: argparse.ArgumentParser, all_help: str) -> None:
        p.add_argument("--section", type=int, help="1-based section number (see 'list')")
        p.add_argument("--line-start", type=int,
                       help="first line of a line range: 0-based book-wide phrase group index (see 'list')")
        p.add_argument("--line-end", type=int, help="last line of the line range (0-based, inclusive)")
        p.add_argument("--all", action="store_true", help=all_help)

    show_llm_fields_help = "also print roster aliases and descriptions (LLM-facing; may contain spoilers)"

    p_roster = sub.add_parser("roster", help="full-book pass: build and cache the character roster")
    add_common(p_roster)
    p_roster.add_argument("--show-llm-fields", action="store_true", help=show_llm_fields_help)
    p_roster.set_defaults(func=command_roster)

    p_list = sub.add_parser("list", help="list sections and the cached roster")
    p_list.add_argument("project_dir")
    p_list.add_argument("--show-llm-fields", action="store_true", help=show_llm_fields_help)
    p_list.set_defaults(func=command_list)

    p_counts = sub.add_parser("counts", help="speaker id counts from saved attributions, descending")
    p_counts.add_argument("project_dir")
    p_counts.add_argument("--lines", metavar="A-B",
                          help="1-based, inclusive line range (default: all lines)")
    p_counts.set_defaults(func=command_counts)

    p_assign = sub.add_parser("assign", help="attribute dialog to roster characters (no project changes)")
    add_common(p_assign)
    add_range_args(p_assign, "the whole book, in chunks, in reading order (resumable)")
    p_assign.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS,
                          help=f"--all: target maximum words per chunk (default {DEFAULT_CHUNK_WORDS})")
    p_assign.add_argument("--plan", action="store_true", help="--all: print the chunk plan and exit")
    p_assign.set_defaults(func=command_assign)

    p_apply = sub.add_parser("apply", help="write voice indices from saved attributions (no LLM call)")
    p_apply.add_argument("project_dir")
    add_range_args(p_apply, "every saved range")
    p_apply.add_argument(
        "--characters", required=True, metavar='"ID=N, ..., REST=N"',
        help="voice mapping: roster ids to 1-based UI voice numbers; REST = everything else spoken"
             f" (unlisted characters, OTHER, UNKNOWN; default voice {DEFAULT_REST_VOICE})."
             " NOT_SPEECH lines get voice 1 (narration).")
    p_apply.set_defaults(func=command_apply)

    args = parser.parse_args()
    if getattr(args, "all", False) and (
            args.section is not None or args.line_start is not None or args.line_end is not None):
        parser.error("--all can't be combined with --section/--line-start/--line-end")
    # LlmSession logs through the app logger, which must be initialized first
    # (log file goes to the temp dir).
    L.init("speaker_attribution")
    # Commands that call the LLM end with the total elapsed time (also when
    # they fail or are interrupted); dry runs that make no LLM call don't.
    uses_llm = args.command in ("roster", "assign") and not (
        args.dump_prompt or args.from_response or getattr(args, "plan", False))
    started = datetime.datetime.now()
    try:
        args.func(args)
    except RuntimeError as error:
        sys.exit(f"Error: {error}")
    finally:
        if uses_llm:
            print(f"\nTotal elapsed: {format_elapsed((datetime.datetime.now() - started).total_seconds())}")


def format_elapsed(seconds: float) -> str:
    """eg "48s", "12m 05s", "1h 02m 09s"."""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
