"""
Prototype: LLM-based speaker attribution for dialog lines.

Builds on the app's existing dialog segmentation (a project imported with
"dialog segmentation" enabled has every dialog phrase group preassigned
voice_index 1). Nothing here re-runs the dialog detector; the stored
project text is the source of truth.

The core logic lives in the app package tts_audiobook_tool/speaker_assign
(roster and attribution types and files, utterances, chunk planning, prompts,
reply parsing). This script adds the LLM calls, the whole-book loop, voice
mapping/application and the command line. Results are saved in
[project]/speaker_assign/: roster.json and attributions.json.

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
    #    --all processes the whole book in chunks (ChunkPlanUtil), in reading
    #    order, one fresh LLM session per chunk; the context before each chunk
    #    carries already-attributed speakers. Resumable: re-running skips
    #    chunks with valid saved results (and retries failed ones).
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
                            (Internally and in saved files, ranges are
                            half-open: [A, B + 1).)
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
from collections import Counter
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time

import tts_audiobook_tool.app_support  # noqa: F401  # pyright: ignore[reportUnusedImport]  (import order: avoids prefs circular import)
from tts_audiobook_tool.l import L
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.app_types.book_serialization import (
    BOOK_FORMAT,
    load_book_from_project_text_file,
)
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.attribution_prompt_util import (
    ATTRIBUTION_SYSTEM_PROMPT,
    AttributionPromptUtil,
)
from tts_audiobook_tool.speaker_assign.attribution_reply_util import AttributionReplyUtil, ParsedReply
from tts_audiobook_tool.speaker_assign.chunk_plan_util import DEFAULT_CHUNK_WORDS, ChunkPlanUtil
from tts_audiobook_tool.speaker_assign.roster import (
    NarrativePerson,
    Prominence,
    Roster,
    RosterCharacter,
    RosterFile,
    RosterNarration,
    name_key,
)
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, RESERVED_SPEAKER_IDS, Utterance
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


PROJECT_TEXT_FILE_NAME = "project_text.json"

# 'apply' voice mapping: NOT_SPEECH lines go to the narration voice; REST (in
# --characters) is everything spoken that isn't listed, defaulting to voice 2
# (the voice the app's dialog detection preassigns).
NOT_SPEECH_VOICE_INDEX = 0
REST_KEYWORD = "REST"
DEFAULT_REST_VOICE = 2  # 1-based UI voice number

CHUNK_RETRY_DELAYS = (10, 30)  # seconds before the 2nd and 3rd attempt at a failed chunk


# ---------------------------------------------------------------------------
# Book, ranges and saved artifacts

def load_book(project_dir: str) -> Book:
    path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
    result = load_book_from_project_text_file(path)
    if isinstance(result, str):
        sys.exit(result)
    return result


def resolve_range(args: argparse.Namespace, book: Book) -> tuple[int, int]:
    """--section N (1-based) or --line-start/--line-end (0-based, inclusive),
    as a half-open range."""
    has_lines = args.line_start is not None or args.line_end is not None
    if args.section is not None and has_lines:
        sys.exit("Use either --section or --line-start/--line-end, not both.")
    if args.section is not None:
        if not 1 <= args.section <= len(book.sections):
            sys.exit(f"--section must be 1..{len(book.sections)}")
        return book.section_ranges()[args.section - 1]
    if args.line_start is None or args.line_end is None:
        sys.exit("Give --section N, or both --line-start and --line-end.")
    last_line = len(book.flat_phrase_groups) - 1
    if not 0 <= args.line_start <= args.line_end <= last_line:
        sys.exit(f"Line range must satisfy 0 <= --line-start <= --line-end <= {last_line}")
    return args.line_start, args.line_end + 1


def range_args_text(args: argparse.Namespace) -> str:
    """The range arguments as given, for suggesting a follow-up command."""
    if args.section is not None:
        return f"--section {args.section}"
    return f"--line-start {args.line_start} --line-end {args.line_end}"


def lines_text(start: int, end: int) -> str:
    """A half-open range as shown to the user: 0-based, inclusive."""
    return f"{start}-{end - 1}"


def require_roster(project_dir: str) -> RosterFile:
    """The saved roster, or exit if there is none (or it can't be loaded)."""
    if not SpeakerAssignFileUtil.has_roster(project_dir):
        sys.exit(f"No speaker roster at {SpeakerAssignFileUtil.roster_path(project_dir)}."
                 " Run the 'roster' step first.")
    result = SpeakerAssignFileUtil.load_roster_file(project_dir)
    if isinstance(result, str):
        sys.exit(result)
    return result


def load_attributions(project_dir: str) -> AttributionsFile:
    result = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(result, str):
        sys.exit(result)
    return result


def save_attributions(project_dir: str, attributions: AttributionsFile) -> None:
    error = SpeakerAssignFileUtil.save_attributions_file(project_dir, attributions)
    if error:
        sys.exit(error)
    print(f"Saved {SpeakerAssignFileUtil.attributions_path(project_dir)}")


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
    for section_index, (start, end) in enumerate(book.section_ranges()):
        sections.append(
            f"===== {UtteranceUtil.section_label(book, section_index)} =====\n"
            + UtteranceUtil.annotate_range(book, start, end, lambda _n, _u: "<q>", section_headers=False).strip()
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


def normalize_roster(value: dict) -> Roster:
    """Build a valid Roster from the LLM's reply, printing what was adjusted.
    Ids: uppercase, no spaces or punctuation (runs of underscores collapse to
    one), not reserved, unique. An entry whose id duplicates an earlier one is
    merged into it when it looks like the same character (see
    same_character), otherwise kept under a suffixed id (GUARD_2). At most one
    character is the narrator. (To be ported as RosterUtil.normalize_roster.)"""
    raw_characters = value.get("characters")
    if not isinstance(raw_characters, list) or not raw_characters:
        raise ValueError("Roster has no 'characters' list")
    characters: list[RosterCharacter] = []
    index_by_id: dict[str, int] = {}
    for raw in raw_characters:
        if not isinstance(raw, dict):
            continue
        char_id = re.sub(r"[^A-Z0-9_]+", "_", str(raw.get("id", "")).upper())
        char_id = re.sub(r"_+", "_", char_id).strip("_")
        if not char_id or char_id in RESERVED_SPEAKER_IDS:
            print(f"  (skipping roster entry with bad id: {raw.get('id')!r})")
            continue
        aliases = raw.get("aliases")
        character = RosterCharacter(
            id=char_id,
            name=str(raw.get("name", char_id)),
            aliases=tuple(str(a) for a in aliases) if isinstance(aliases, list) else (),
            gender=str(raw.get("gender", "unknown")),
            description=str(raw.get("description", "")),
            is_narrator=bool(raw.get("is_narrator")),
            prominence=Prominence.from_value(raw.get("prominence")),
        )
        existing_index = index_by_id.get(char_id)
        if existing_index is not None and same_character(characters[existing_index], character):
            characters[existing_index] = merge_character(characters[existing_index], character)
            print(f"  (merged duplicate roster entry {char_id}: {character.name!r})")
            continue
        if existing_index is not None:
            suffix = 2
            while f"{char_id}_{suffix}" in index_by_id:
                suffix += 1
            character = dataclasses.replace(character, id=f"{char_id}_{suffix}")
            print(f"  (duplicate roster id {char_id} for a different character"
                  f" {character.name!r}; renamed {character.id})")
        index_by_id[character.id] = len(characters)
        characters.append(character)

    narrator_seen = False
    for index, character in enumerate(characters):
        if character.is_narrator and narrator_seen:
            characters[index] = dataclasses.replace(character, is_narrator=False)
        narrator_seen = narrator_seen or character.is_narrator

    raw_narration = value.get("narration")
    narration = raw_narration if isinstance(raw_narration, dict) else {}
    return Roster(
        characters=tuple(characters),
        narration=RosterNarration(
            person=NarrativePerson.from_value(narration.get("person")),
            notes=str(narration.get("notes", "")),
        ),
    )


def same_character(a: RosterCharacter, b: RosterCharacter) -> bool:
    """Same name, or one's name among the other's aliases. (Alias-to-alias
    overlap isn't enough: generic aliases like "madame" are often shared.)"""
    name_a, name_b = name_key(a.name), name_key(b.name)
    return (name_a == name_b
            or name_a in {name_key(alias) for alias in b.aliases}
            or name_b in {name_key(alias) for alias in a.aliases})


def merge_character(target: RosterCharacter, other: RosterCharacter) -> RosterCharacter:
    """A duplicate entry folded into the first one: union of names/aliases, the
    first non-empty description and known gender, the higher prominence."""
    aliases = list(target.aliases)
    known = target.name_keys()
    for alias in (other.name, *other.aliases):
        if name_key(alias) not in known:
            aliases.append(alias)
            known.add(name_key(alias))
    return dataclasses.replace(
        target,
        aliases=tuple(aliases),
        description=target.description or other.description,
        gender=other.gender if target.gender == "unknown" else target.gender,
        prominence=max(target.prominence, other.prominence, key=lambda p: p.rank),
        is_narrator=target.is_narrator or other.is_narrator,
    )


def command_roster(args: argparse.Namespace) -> None:
    book = load_book(args.project_dir)
    if not book.segmentation_settings.dialog_segmentation:
        print("Warning: this book was not segmented with dialog segmentation enabled;"
              " dialog lines may not be marked.")
    total_utterances = len(UtteranceUtil.find_utterances(book, 0, len(book.flat_phrase_groups)))
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
    roster_file = RosterFile(
        roster=normalize_roster(AttributionReplyUtil.extract_json(reply)),
        created=now_iso(),
        llm=llm_description(),
        book_title=book.title,
        section_text_hashes=UtteranceUtil.section_text_hashes(book),
        raw_reply=reply,
    )
    error = SpeakerAssignFileUtil.save_roster_file(args.project_dir, roster_file)
    if error:
        sys.exit(error)
    print(f"Saved {SpeakerAssignFileUtil.roster_path(args.project_dir)}")
    # A new roster invalidates attributions made against the previous one.
    attributions_path = SpeakerAssignFileUtil.attributions_path(args.project_dir)
    if os.path.exists(attributions_path):
        error = SpeakerAssignFileUtil.delete_attributions_file(args.project_dir)
        if error:
            sys.exit(error)
        print(f"Removed saved attributions (they used the previous roster): {attributions_path}")
    print_roster(roster_file.roster, args.show_llm_fields)


def print_roster(roster: Roster, show_llm_fields: bool) -> None:
    """Print the roster's user-facing (spoiler-free) fields; descriptions and
    aliases are LLM-facing and may contain spoilers, so they're shown only
    when show_llm_fields is set."""
    narration = roster.narration
    if narration != RosterNarration():
        print(f"\nNarration: {narration.person.value or '?'} person. {narration.notes}")
    print(f"\n{'#':>3}  {'id':<20} {'tier':<10} {'gender':<7} name")
    for i, character in enumerate(roster.characters, 1):
        narrator = " [NARRATOR]" if character.is_narrator else ""
        tier = character.prominence.value or "?"
        print(f"{i:>3}  {character.id:<20} {tier:<10} {character.gender:<7} {character.name}{narrator}")
        if show_llm_fields:
            indent = " " * 43
            if character.aliases:
                print(f"{indent} aliases: {', '.join(character.aliases)}")
            if character.description:
                print(f"{indent} {character.description}")


# ---------------------------------------------------------------------------
# list, counts

def command_list(args: argparse.Namespace) -> None:
    book = load_book(args.project_dir)
    attributions = load_attributions(args.project_dir)
    saved = [(r.start, r.end, len(r.utterances)) for r in attributions.ranges]
    # assigned: "yes" = saved results for exactly this section, "part" = a
    # saved line range overlaps it
    print(f"{'sect':>4}  {'lines':<11}  {'utter.':>6}  {'assigned':<8} title")
    for i, (start, end) in enumerate(book.section_ranges()):
        assigned = ""
        if any((s, e) == (start, end) for s, e, _ in saved):
            assigned = "yes"
        elif any(s < end and e > start for s, e, _ in saved):
            assigned = "part"
        print(f"{i + 1:>4}  {lines_text(start, end):<11}  "
              f"{len(UtteranceUtil.find_utterances(book, start, end)):>6}  {assigned:<8} "
              f"{book.sections[i].title.strip()}")
    if saved:
        print("\nSaved line ranges: " + ", ".join(
            f"{lines_text(s, e)} ({n} utterances)" for s, e, n in sorted(saved)))
    if SpeakerAssignFileUtil.has_roster(args.project_dir):
        print_roster(require_roster(args.project_dir).roster, args.show_llm_fields)


def command_counts(args: argparse.Namespace) -> None:
    """Speaker id counts from saved attributions, for --lines A-B (1-based,
    inclusive) or the whole book."""
    book = load_book(args.project_dir)
    attributions = load_attributions(args.project_dir)
    line_count = len(book.flat_phrase_groups)
    if args.lines:
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", args.lines)
        if not match:
            sys.exit("--lines must look like A-B, eg 252-481 (1-based, inclusive)")
        first, last = int(match[1]), int(match[2])
        if not 1 <= first <= last <= line_count:
            sys.exit(f"--lines must satisfy 1 <= A <= B <= {line_count}")
        start, end = first - 1, last
        title = f"Speaker counts, lines {first}-{last}"
    else:
        start, end = 0, line_count
        title = "Speaker counts, all lines"
    counts, unattributed = speaker_counts(attributions, book, start, end)
    print_speaker_counts(title, counts, unattributed)


def counts_text(counts: Counter[str]) -> str:
    return ", ".join(f"{label} {count}" for label, count in counts.most_common())


def speaker_counts(attributions: AttributionsFile, book: Book, start: int, end: int) -> tuple[Counter[str], int]:
    """Speaker id counts over saved, still-valid utterances whose first line is
    within [start, end), and the number of dialog utterances there that have
    no saved result."""
    speakers = attributions.speakers_by_line(book)
    counts = Counter(SpeakerLabelUtil.counting_id(label)
                     for line, label in speakers.items() if start <= line < end)
    unattributed = sum(1 for u in UtteranceUtil.find_utterances(book, start, end) if u.first_line not in speakers)
    return counts, unattributed


def print_speaker_counts(title: str, counts: Counter[str], unattributed: int = 0) -> None:
    """Speaker ids by utterance count, descending."""
    pending = f", {unattributed:,} not attributed yet" if unattributed else ""
    print(f"\n{title}: {sum(counts.values()):,} attributed utterances{pending}")
    for speaker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count:>6,}  {speaker}")


# ---------------------------------------------------------------------------
# Step 2: assign (attribute each dialog utterance to a roster character)

def attribute(
        message: str, utterances: list[Utterance], roster_ids: frozenset[str], reply: str | None = None,
) -> ParsedReply:
    """A speaker for every utterance (ids 1..n): one fresh LLM session (or a
    saved reply), plus one follow-up turn for ids missing from the reply.
    Ids still missing end up UNKNOWN (see make_range_attribution)."""
    requested = set(range(1, len(utterances) + 1))
    llm = None
    if reply is None:
        llm = make_llm(ATTRIBUTION_SYSTEM_PROMPT)
        reply = send_streaming(llm, message)
    parsed = AttributionReplyUtil.parse_reply(AttributionReplyUtil.extract_json(reply), requested, roster_ids)
    if parsed.missing and llm is not None:
        print(f"  {len(parsed.missing)} id(s) missing from reply; asking again for those")
        follow_up = send_streaming(llm, AttributionPromptUtil.missing_ids_message(parsed.missing))
        parsed = AttributionReplyUtil.merge(parsed, AttributionReplyUtil.parse_reply(
            AttributionReplyUtil.extract_json(follow_up), set(parsed.missing), roster_ids))
    if parsed.missing:
        print(f"  {len(parsed.missing)} id(s) still missing; marked UNKNOWN")
    return parsed


def attribute_with_retries(
        message: str, utterances: list[Utterance], roster_ids: frozenset[str],
) -> ParsedReply | str:
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


def warn_if_roster_stale(roster_file: RosterFile, book: Book, section_indices: range) -> None:
    saved, current = roster_file.section_text_hashes, UtteranceUtil.section_text_hashes(book)
    if len(saved) != len(current) or any(saved[i] != current[i] for i in section_indices):
        print("Warning: project text changed since the roster pass "
              "(re-run 'roster' if the change is substantial).")


def command_assign(args: argparse.Namespace) -> None:
    book = load_book(args.project_dir)
    if args.all and args.plan:  # the chunk plan doesn't need the roster
        chunks = ChunkPlanUtil.plan_chunks(book, args.chunk_words)
        print(f"{book.title or '(untitled)'}: {len(chunks)} chunks of up to {args.chunk_words:,} words")
        print_plan(book, chunks)
        return
    roster_file = require_roster(args.project_dir)
    attributions = load_attributions(args.project_dir)
    if args.all:
        warn_if_roster_stale(roster_file, book, range(len(book.sections)))
        assign_all(args, book, roster_file.roster, attributions)
        return

    start, end = resolve_range(args, book)
    warn_if_roster_stale(roster_file, book, range(
        UtteranceUtil.section_of_line(book, start), UtteranceUtil.section_of_line(book, end - 1) + 1))
    utterances = UtteranceUtil.find_utterances(book, start, end)
    print(f"{UtteranceUtil.range_label(book, start, end)}: {len(utterances)} utterances")
    if not utterances:
        return

    roster = roster_file.roster
    message = AttributionPromptUtil.build_message(book, start, end, roster, attributions.speakers_by_line(book))
    if maybe_dump_prompt(args, ATTRIBUTION_SYSTEM_PROMPT, message):
        return
    parsed = attribute(message, utterances, roster.ids, read_saved_response(args))
    result = AttributionReplyUtil.make_range_attribution(
        book, start, end, utterances, parsed, created=now_iso(), llm=llm_description())
    print_review(result.utterances)

    save_attributions(args.project_dir, attributions.with_range(result))
    print(f'\nApply with: apply {args.project_dir} {range_args_text(args)} --characters "ID=N, ..., REST=N"')


def assign_all(args: argparse.Namespace, book: Book, roster: Roster, attributions: AttributionsFile) -> None:
    """Attribute the whole book, chunk by chunk in reading order, one fresh
    LLM session per chunk. Results are saved after every chunk; chunks that
    already have valid saved results are skipped, so an interrupted or
    partly failed run continues where it left off when re-run."""
    if args.dump_prompt or args.from_response:
        sys.exit("--dump-prompt/--from-response work with --section or a line range, not --all")
    chunks = ChunkPlanUtil.plan_chunks(book, args.chunk_words)
    print(f"{book.title or '(untitled)'}: {len(chunks)} chunks of up to {args.chunk_words:,} words")

    failures: list[str] = []
    for number, (start, end) in enumerate(chunks, 1):
        utterances = UtteranceUtil.find_utterances(book, start, end)
        heading = (f"\n[{number}/{len(chunks)}] {UtteranceUtil.range_label(book, start, end)}:"
                   f" {len(utterances)} utterances")
        if not utterances:
            print(heading + " (no dialog; skipped)")
            continue
        saved = attributions.get(start, end)
        if saved is not None and saved.is_valid_for(book):
            print(heading + " (already done)")
            continue
        print(heading)
        message = AttributionPromptUtil.build_message(book, start, end, roster, attributions.speakers_by_line(book))
        parsed = attribute_with_retries(message, utterances, roster.ids)
        if isinstance(parsed, str):  # keep going; failed chunks are retried by the next run
            failures.append(f"lines {lines_text(start, end)}: {parsed}")
            print(f"  FAILED: {parsed}")
            continue
        result = AttributionReplyUtil.make_range_attribution(
            book, start, end, utterances, parsed, created=now_iso(), llm=llm_description())
        attributions = attributions.with_range(result)
        save_attributions(args.project_dir, attributions)
        print("  " + counts_text(Counter(SpeakerLabelUtil.counting_id(u.speaker) for u in result.utterances)))
    print_book_summary(attributions, book, failures)


def print_book_summary(attributions: AttributionsFile, book: Book, failures: list[str]) -> None:
    print("\n===== Summary =====")
    counts, unattributed = speaker_counts(attributions, book, 0, len(book.flat_phrase_groups))
    print_speaker_counts("Speaker counts, all lines", counts, unattributed)
    current = list(attributions.by_first_line(book).values())
    others = Counter(name for u in current if (name := SpeakerLabelUtil.other_name(u.speaker)))
    recurring = Counter({name: count for name, count in others.items() if count >= 3})
    print(f"\nFlagged uncertain: {sum(1 for u in current if u.uncertain)}")
    if recurring:
        print(f"Recurring OTHER speakers (roster candidates): {counts_text(recurring)}")
    if failures:
        print(f"\nFailed chunks ({len(failures)}); re-run the same command to retry them:")
        for failure in failures:
            print(f"  {failure}")


def print_plan(book: Book, chunks: list[tuple[int, int]]) -> None:
    print(f"\n{'#':>3}  {'lines':<11} {'words':>6} {'utter.':>6}  range")
    for number, (start, end) in enumerate(chunks, 1):
        words = sum(len(group.text.split()) for group in book.flat_phrase_groups[start:end])
        print(f"{number:>3}  {lines_text(start, end):<11} {words:>6}"
              f" {len(UtteranceUtil.find_utterances(book, start, end)):>6}"
              f"  {UtteranceUtil.range_label(book, start, end)}")


# ---------------------------------------------------------------------------
# Step 3: apply (map speakers to voices and write voice indices)

@dataclass
class VoiceMapping:
    """Speaker label -> zero-based voice_index. `by_id` holds the listed roster
    characters; everything else spoken (unlisted characters, OTHER, UNKNOWN)
    gets `rest`. NOT_SPEECH always gets the narration voice."""
    by_id: dict[str, int]
    rest: int

    def index_for(self, label: str) -> int:
        if label == NOT_SPEECH:
            return NOT_SPEECH_VOICE_INDEX
        return self.by_id.get(label, self.rest)


def parse_voice_mapping(text: str, roster: Roster) -> VoiceMapping:
    """Parse --characters, eg "BARON=2, DIDIER=3, REST=4": roster ids (or the
    keyword REST) mapped to 1-based UI voice numbers. REST defaults to voice
    DEFAULT_REST_VOICE."""
    known = set(roster.ids)
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
        roster: Roster, mapping: VoiceMapping, project_dir: str, voice_counts: Counter[int],
) -> None:
    """Per voice: who it gets and how many utterances."""
    names = {c.id: c.name for c in roster.characters}
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
        line_voices: Sequence[tuple[Sequence[int], int]],
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
    for lines, new_index in line_voices:
        for line in lines:
            if raw_groups[line].get("voice_index") != new_index:
                raw_groups[line]["voice_index"] = new_index
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
    or with --all every saved range, newer results winning where ranges
    overlap. The voice mapping (--characters) is chosen here."""
    book = load_book(args.project_dir)
    roster = require_roster(args.project_dir).roster
    mapping = parse_voice_mapping(args.characters, roster)
    attributions = load_attributions(args.project_dir)
    if args.all:
        for r in attributions.ranges:
            if not r.is_valid_for(book):
                print(f"Skipping lines {lines_text(r.start, r.end)}: project text changed since it was attributed")
        utterances = list(attributions.by_first_line(book).values())
        if not utterances:
            sys.exit("No saved attributions to apply; run 'assign' first.")
        print(f"Applying {len(attributions.valid_ranges(book))} saved range(s), {len(utterances):,} utterances")
    else:
        start, end = resolve_range(args, book)
        saved = attributions.get(start, end)
        if saved is None:
            saved_ranges = ", ".join(lines_text(r.start, r.end) for r in attributions.ranges) or "none"
            sys.exit(f"No saved attributions for lines {lines_text(start, end)}; run 'assign' over that range"
                     f" first. (Saved ranges: {saved_ranges})")
        if not saved.is_valid_for(book):
            sys.exit("Project text of this range changed since it was attributed; re-run 'assign'.")
        utterances = list(saved.utterances)
        print(f"{UtteranceUtil.range_label(book, start, end)}: applying {len(utterances)} saved attributions"
              f" from {saved.created or '?'}")
        print_review(utterances, mapping)

    line_voices = [(u.lines, mapping.index_for(u.speaker)) for u in utterances]
    voice_counts: Counter[int] = Counter(voice for _, voice in line_voices)
    print_voice_map(roster, mapping, args.project_dir, voice_counts)
    write_voices(args.project_dir, book, line_voices)


def print_review(utterances: Sequence[UtteranceAttribution], mapping: VoiceMapping | None = None) -> None:
    """One row per utterance (numbered by position); the voice column only
    when a voice mapping is given."""
    voice_header = f"{'voice':>5}  " if mapping is not None else ""
    print(f"\n{'#':>4}  {voice_header}{'speaker':<22} text")
    counts: Counter[str] = Counter()
    for number, utterance in enumerate(utterances, 1):
        counts[SpeakerLabelUtil.counting_id(utterance.speaker)] += 1
        voice = f"{mapping.index_for(utterance.speaker) + 1:>5}  " if mapping is not None else ""
        flag = "?" if utterance.uncertain else " "
        excerpt = " ".join(utterance.text.split())
        if len(excerpt) > 70:
            excerpt = excerpt[:67] + "..."
        print(f"{number:>4}{flag} {voice}{utterance.speaker[:22]:<22} {excerpt}")
    print_speaker_counts("Speaker counts", counts)
    uncertain = sum(1 for u in utterances if u.uncertain)
    if uncertain:
        print(f"Flagged uncertain: {uncertain} (marked '?')")


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

    p_roster = sub.add_parser("roster", help="full-book pass: build and save the character roster")
    add_common(p_roster)
    p_roster.add_argument("--show-llm-fields", action="store_true", help=show_llm_fields_help)
    p_roster.set_defaults(func=command_roster)

    p_list = sub.add_parser("list", help="list sections and the saved roster")
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
