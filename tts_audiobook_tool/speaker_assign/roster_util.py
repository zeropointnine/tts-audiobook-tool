from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import re
from typing import Any

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.roster import (
    NarrativePerson,
    Prominence,
    Roster,
    RosterCharacter,
    RosterNarration,
    name_key,
)
from tts_audiobook_tool.speaker_assign.speaker_assign_types import RESERVED_SPEAKER_IDS
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


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


@dataclass(frozen=True)
class NormalizedRoster:
    """A roster built from the LLM's reply, plus human-readable notes on what
    was adjusted (dropped, merged or renamed entries)."""
    roster: Roster
    notes: tuple[str, ...] = ()


class RosterUtil:
    """The full-book roster request and the normalization of its reply."""

    @staticmethod
    def build_message(book: Book) -> str:
        """The roster request's user message: instructions and the whole book,
        sections headed, every utterance wrapped in <q>…</q>."""
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

    @staticmethod
    def normalize_roster(value: dict[str, Any]) -> NormalizedRoster:
        """
        A valid Roster from the LLM's reply JSON. Ids: uppercase, no spaces or
        punctuation (runs of underscores collapse to one), not reserved,
        unique. An entry whose id duplicates an earlier one is merged into it
        when it looks like the same character (see same_character), otherwise
        kept under a suffixed id (GUARD_2). At most one character is the
        narrator. Raises ValueError if the reply has no usable characters.
        """
        raw_characters = value.get("characters")
        if not isinstance(raw_characters, list) or not raw_characters:
            raise ValueError("Roster has no 'characters' list")
        notes: list[str] = []
        characters: list[RosterCharacter] = []
        index_by_id: dict[str, int] = {}
        for raw in raw_characters:
            if not isinstance(raw, dict):
                continue
            char_id = re.sub(r"[^A-Z0-9_]+", "_", str(raw.get("id", "")).upper())
            char_id = re.sub(r"_+", "_", char_id).strip("_")
            if not char_id or char_id in RESERVED_SPEAKER_IDS:
                notes.append(f"Skipped roster entry with bad id: {raw.get('id')!r}")
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
            if existing_index is not None and RosterUtil.same_character(characters[existing_index], character):
                characters[existing_index] = RosterUtil.merge_character(characters[existing_index], character)
                notes.append(f"Merged duplicate roster entry {char_id}: {character.name!r}")
                continue
            if existing_index is not None:
                suffix = 2
                while f"{char_id}_{suffix}" in index_by_id:
                    suffix += 1
                character = dataclasses.replace(character, id=f"{char_id}_{suffix}")
                notes.append(f"Duplicate roster id {char_id} for a different character"
                             f" {character.name!r}; renamed {character.id}")
            index_by_id[character.id] = len(characters)
            characters.append(character)

        if not characters:
            raise ValueError("Roster has no characters with usable ids")

        narrator_seen = False
        for index, character in enumerate(characters):
            if character.is_narrator and narrator_seen:
                characters[index] = dataclasses.replace(character, is_narrator=False)
            narrator_seen = narrator_seen or character.is_narrator

        raw_narration = value.get("narration")
        narration = raw_narration if isinstance(raw_narration, dict) else {}
        roster = Roster(
            characters=tuple(characters),
            narration=RosterNarration(
                person=NarrativePerson.from_value(narration.get("person")),
                notes=str(narration.get("notes", "")),
            ),
        )
        return NormalizedRoster(roster, tuple(notes))

    @staticmethod
    def same_character(a: RosterCharacter, b: RosterCharacter) -> bool:
        """Same name, or one's name among the other's aliases. (Alias-to-alias
        overlap isn't enough: generic aliases like "madame" are often shared.)"""
        name_a, name_b = name_key(a.name), name_key(b.name)
        return (name_a == name_b
                or name_a in {name_key(alias) for alias in b.aliases}
                or name_b in {name_key(alias) for alias in a.aliases})

    @staticmethod
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

    @staticmethod
    def is_stale(section_text_hashes: tuple[str, ...], book: Book) -> bool:
        """True if the book's text changed since the roster was made from it
        (per-section text hashes differ, or the section count does)."""
        return section_text_hashes != UtteranceUtil.section_text_hashes(book)
