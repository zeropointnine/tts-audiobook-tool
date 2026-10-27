"""Optional "Prep voices list": appends a named, sample-less project voice for each
chosen character, and maps the character to it for step 3 (see voices_menu)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.menus.speaker_assign.voices_menu import sort_characters
from tts_audiobook_tool.project_support.project_voice_util import VOICE_NAME_MAX_LENGTH
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.attribution_summary import AttributionSummary
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.util import make_noun, print_feedback, printt


SEED_VOICES_LABEL = "Prep voices list"


def has_attribution_data(project_dir: str) -> bool:
    """Whether step 2 has assigned any dialog (the menu item's visibility)."""
    if not SpeakerAssignFileUtil.has_roster(project_dir):
        return False
    attributions = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    return isinstance(attributions, AttributionsFile) and bool(attributions.ranges)


def seed_voices_menu(state: State) -> None:

    project = state.project
    project_dir = project.dir_path
    roster_file = SpeakerAssignFileUtil.load_roster_file(project_dir)
    if isinstance(roster_file, str):
        ask.ask_error(roster_file)
        return
    attributions = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(attributions, str):
        ask.ask_error(attributions)
        return
    voice_map = SpeakerAssignFileUtil.load_voices_file(project_dir)
    if isinstance(voice_map, str):
        # Not treated as empty: saving would overwrite the file
        ask.ask_error(voice_map)
        return

    # The same list as step 3: characters with dialog, most passages first
    counts = AttributionSummary.make(attributions, project.book).counts
    ids = [character.id for character in sort_characters(roster_file.roster.characters, counts, omit_unused=True)]
    if not ids:
        print_feedback("No dialog has been assigned to characters yet (step 2)", is_error=True)
        return

    subheading = SEED_VOICES_DESC + "\n\n" + "\n".join(make_numbered_columns(ids, counts))
    MenuUtil.print_screen_heading(state, "Prep voices list", subheading=subheading, breadcrumb="Prep voices list")
    printt(f'Enter numbers {COL_DIM}(eg, "1-3, 5, 8" or "all")')
    inp = ask.ask_input()
    if not inp:
        return
    # Lenient: out-of-range numbers are dropped; only syntax errors are reported
    indices, errors = RangeStringUtil.parse_ranges_string(inp, len(ids), strict=False)
    if errors:
        print_feedback(", ".join(errors), is_error=True)
        return

    # Everything is worked out (and validated) before anything is saved
    plan = SeedPlan.make(project.voice_references, [ids[index] for index in sorted(indices)])
    if plan.added:
        new_voice_map = plan.make_voice_map(voice_map, len(project.voice_references))
        previous = project.voice_references
        project.voice_references = previous + plan.make_voices()
        error = project.save()
        if error:
            project.voice_references = previous
            ask.ask_error(error)
            return
        error = SpeakerAssignFileUtil.save_voices_file(project_dir, new_voice_map)
        if error:
            ask.ask_error(f"The voices were added, but mapping them to their characters failed: {error}")
            return
    ask.ask_enter_to_continue(plan.make_message())


@dataclass(frozen=True)
class SeedPlan:
    """The voices to append: (character id, voice name) pairs, in list order,
    plus the number of chosen characters skipped because a voice already has
    that name (case-insensitive)."""
    added: tuple[tuple[str, str], ...]
    skipped: int

    @staticmethod
    def make(voices: Sequence[Voice], character_ids: Sequence[str]) -> SeedPlan:
        # Ids longer than the voice-name limit are truncated; a truncated name
        # that collides is skipped like any other existing name
        taken = {voice.name.strip().casefold() for voice in voices if voice.name.strip()}
        added = []
        skipped = 0
        for character_id in character_ids:
            name = character_id[:VOICE_NAME_MAX_LENGTH]
            key = name.casefold()
            if key in taken:
                skipped += 1
                continue
            taken.add(key)
            added.append((character_id, name))
        return SeedPlan(tuple(added), skipped)

    def make_voices(self) -> list[Voice]:
        return [Voice(name=name) for _, name in self.added]

    def make_voice_map(self, voice_map: VoiceMapFile, first_index: int) -> VoiceMapFile:
        """`voice_map` with each added character mapped to its new voice, which
        is appended at `first_index` onward."""
        for offset, (character_id, _) in enumerate(self.added):
            voice_map = voice_map.with_voice(character_id, first_index + offset)
        return voice_map

    def make_message(self) -> str:
        count = len(self.added)
        if count:
            message = (f"Added {count:,} {make_noun('item', 'items', count)}, "
                       f"mapped to {make_noun('its character', 'their characters', count)} in step 3.")
        else:
            message = "Added 0 items."
        if self.skipped:
            message += (f" Skipped {self.skipped:,} "
                        f"{make_noun('item which already exists', 'items which already exist', self.skipped)}.")
        return message


def make_numbered_columns(ids: Sequence[str], counts: Mapping[str, int]) -> list[str]:
    """Eg "[1] DUDE (55)" per line, the dialog passage count dimmed; past
    COLUMN_THRESHOLD items, two columns (numbered down the first, then down
    the second)."""
    # (visible text, colored text) per entry; padding goes by the visible length
    entries = [
        (f"[{number}] {character_id} ({counts.get(character_id, 0):,})",
         f"{COL_DEFAULT}[{number}] {character_id} {COL_DIM}({counts.get(character_id, 0):,})")
        for number, character_id in enumerate(ids, 1)
    ]
    if len(entries) <= COLUMN_THRESHOLD:
        return [colored for _, colored in entries]
    rows = (len(entries) + 1) // 2
    left, right = entries[:rows], entries[rows:]
    width = max(len(plain) for plain, _ in left) + COLUMN_GAP
    lines = []
    for row, (plain, colored) in enumerate(left):
        if row < len(right):
            colored += " " * (width - len(plain)) + right[row][1]
        lines.append(colored)
    return lines


COLUMN_THRESHOLD = 10
COLUMN_GAP = 4

SEED_VOICES_DESC = "Appends new (unconfigured) voice items for each item selected."
