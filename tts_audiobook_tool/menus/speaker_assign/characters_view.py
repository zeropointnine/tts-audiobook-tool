"""
The consolidated "character/dialog results" view: the roster (step 1)
combined with the per-character dialog counts from attribution (step 2).
"""
from collections import Counter
from collections.abc import Sequence

from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_DIM_ITALICS, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution_summary import AttributionSummary
from tts_audiobook_tool.speaker_assign.roster import RosterCharacter, RosterNarration
from tts_audiobook_tool.speaker_assign.roster_util import RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import (
    NOT_SPEECH,
    OTHER,
    RESERVED_SPEAKER_IDS,
    RESERVED_SPEAKER_ORDER,
    UNKNOWN,
)
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import printt


RESULTS_HEADING = "Character/dialog results"
VIEW_RESULTS_LABEL = "View character/dialog results"


def make_roster_stale_marker(state: State) -> str:
    """Mark known stale roster data; unreadable data has no known provenance."""
    roster_file = SpeakerAssignFileUtil.load_roster_file(state.project.dir_path)
    if not isinstance(roster_file, str) and RosterUtil.is_stale(roster_file.section_text_hashes, state.project.book):
        return f"{COL_ERROR}*"
    return ""


def characters_menu(state: State) -> None:

    def on_summary(_: State, __: MenuItem) -> None:
        # The breadcrumb names the view, so the screen headings don't repeat it
        MenuUtil.print_screen_heading(state, "Summary", breadcrumb="Summary")
        if print_characters_info(state):
            ask.ask_enter_to_continue()

    def on_expanded(_: State, __: MenuItem) -> None:
        MenuUtil.print_screen_heading(state, "Expanded", breadcrumb="Expanded")
        if print_characters_info(state, expanded=True):
            ask.ask_enter_to_continue()

    MenuUtil.menu(
        state,
        heading=RESULTS_HEADING,
        items=[
            MenuItem("Summary", on_summary),
            MenuItem(f"Expanded {COL_DIM}(possible spoilers)", on_expanded),
        ],
        breadcrumb=RESULTS_HEADING,
    )


def print_characters_info(state: State, expanded: bool = False) -> bool:
    """
    Prints the saved roster combined with the saved attribution results.

    Before step 2 has saved anything, characters are listed in roster order
    with an index; after, by dialog passage count (most first) followed by
    the reserved answers, and the step 2 progress.

    Aliases and descriptions are LLM-facing and may contain spoilers, so
    they're left out unless `expanded`, which prints each character as a
    block that includes them.

    Shows an error and returns False if the saved data can't be loaded.
    """
    project_dir = state.project.dir_path
    book = state.project.book
    roster_file = SpeakerAssignFileUtil.load_roster_file(project_dir)
    if isinstance(roster_file, str):
        ask.ask_error(roster_file)
        return False
    attributions = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(attributions, str):
        ask.ask_error(attributions)
        return False
    roster = roster_file.roster

    has_counts = bool(attributions.ranges)
    summary = AttributionSummary.make(attributions, book) if has_counts else None
    counts: Counter[str] = summary.counts if summary else Counter()

    # Sections are separated by a blank line, without a leading one under the heading
    has_printed = False
    def print_gap() -> None:
        if has_printed:
            printt()

    if RosterUtil.is_stale(roster_file.section_text_hashes, book):
        printt(f"{COL_ERROR}* {COL_DIM_ITALICS}The project text has changed since this roster was created")
        has_printed = True

    narration = roster.narration
    if narration != RosterNarration():
        person = f"{narration.person.value} person" if narration.person.value else "Unknown person"
        print_gap()
        printt(f"Narration: {person}")
        if narration.notes.strip():
            printt(f"{COL_DIM_ITALICS}{narration.notes.strip()}")
        if narration.voice_hint.strip():
            printt(f"{COL_DIM}voice hint: {COL_DEFAULT}{narration.voice_hint.strip()}")
        has_printed = True

    characters: Sequence[RosterCharacter] = roster.characters
    if has_counts:
        # Stable, so equal counts keep roster order
        characters = sorted(characters, key=lambda character: -counts[character.id])
    extra_rows = make_extra_rows(summary, {character.id for character in roster.characters}) if summary else []

    id_width = max(len("id"), *(len(character.id) for character in characters), *(len(id) for id, _, _ in extra_rows))
    # The first column (count, or # before step 2) is as wide as its header or widest value
    if has_counts:
        first_width = max(len("count"), *(len(f"{count:,}") for count in counts.values()))
    else:
        first_width = max(len("#"), len(str(len(characters))))
    print_gap()
    if expanded:
        print_characters_expanded(characters, counts if has_counts else None, id_width)
        if extra_rows:
            printt(f"{COL_DIM}{'count':>{first_width}}  id")
            print_extra_rows(extra_rows, first_width, id_width)
            printt()
    else:
        print_characters_table(characters, counts if has_counts else None, first_width, id_width)
        print_extra_rows(extra_rows, first_width, id_width, tier_gender_gap=True)
        printt()

    if summary:
        plan = AssignPlan.make(book, attributions)
        total = summary.attributed + summary.unattributed
        printt(f"{COL_DIM}{summary.attributed:,} of {total:,} dialog passages assigned "
               f"({plan.done_count} of {len(plan.chunks)} chunks)")
        printt()

    if not expanded:
        path = SpeakerAssignFileUtil.roster_path(project_dir)
        printt(f"{COL_DEFAULT}Full roster data: "
               f"{COL_DIM}{text_util.make_terminal_hyperlink(path, is_file=True)} (may contain spoilers)")
        printt()

    provenance = []
    if roster_file.created:
        provenance.append(f"{COL_DEFAULT}Created: {COL_DIM}{roster_file.created.replace('T', ' ')[:16]}")
    if roster_file.llm.get("model"):
        provenance.append(f"{COL_DEFAULT}model: {COL_DIM}{roster_file.llm['model']}")
    if provenance:
        printt(", ".join(provenance))
        printt()
    return True


# Width of the tier and gender columns, plus their trailing gaps
TIER_WIDTH = 10
GENDER_WIDTH = 7


def print_characters_table(
        characters: Sequence[RosterCharacter], counts: Counter[str] | None, first_width: int, id_width: int,
) -> None:
    """
    One row per character, its voice hint on a second line under the id
    column. The first column is the dialog count, or the roster index when
    there are no counts yet.
    """
    first_column = "count" if counts is not None else "#"
    printt(f"{COL_DIM}{first_column:>{first_width}}  {'id':<{id_width}}  {'tier':<{TIER_WIDTH}}  "
           f"{'gender':<{GENDER_WIDTH}}  name")
    hint_indent = " " * (first_width + 2)
    for i, character in enumerate(characters, 1):
        if counts is None:
            first = f"{COL_DIM}{i:>{first_width}}"
        else:
            count = counts[character.id]
            first = f"{COL_DIM if not count else COL_DEFAULT}{count:>{first_width},}"
        narrator = f" {COL_DIM}[narrator]" if character.is_narrator else ""
        tier = character.prominence.value or "?"
        printt(
            f"{first}  {COL_ACCENT}{character.id:<{id_width}}{COL_DEFAULT}  "
            f"{tier:<{TIER_WIDTH}}  {character.gender:<{GENDER_WIDTH}}  {character.name}{narrator}"
        )
        if character.voice_hint.strip():
            printt(f"{hint_indent}{COL_DIM}voice hint: {character.voice_hint.strip()}")


def print_characters_expanded(
        characters: Sequence[RosterCharacter], counts: Counter[str] | None, id_width: int) -> None:
    """Prints each character as a block, including the spoiler-prone aliases and description."""
    # Index padded to the widest one so the id column and the indent line up
    index_width = len(f"[{len(characters)}]")
    for i, character in enumerate(characters, 1):
        index = f"[{i}]".ljust(index_width)
        indent = " " * (index_width + 1)
        tier = character.prominence.value or "?"
        count = f"  {COL_DIM}count: {COL_DEFAULT}{counts[character.id]:,}" if counts is not None else ""
        narrator = f"  {COL_DIM}[narrator]" if character.is_narrator else ""
        gender = f"{character.gender:<{GENDER_WIDTH}}" if count else character.gender
        printt(
            f"{COL_DIM}{index} id: {COL_ACCENT}{character.id:<{id_width}}  "
            f"{COL_DIM}tier: {COL_DEFAULT}{tier:<{TIER_WIDTH}} "
            f"{COL_DIM}gender: {COL_DEFAULT}{gender}{count}{narrator}"
        )
        # An alias that just repeats the name adds nothing here
        alias_list = [alias for alias in character.aliases if alias.strip() != character.name.strip()]
        aliases = ", ".join(alias_list) if alias_list else f"{COL_DIM}(none)"
        printt(f"{indent}{COL_DIM}name: {COL_DEFAULT}{character.name}")
        printt(f"{indent}{COL_DIM}aliases: {COL_DEFAULT}{aliases}")
        if character.voice_hint.strip():
            printt(f"{indent}{COL_DIM}voice hint: {COL_DEFAULT}{character.voice_hint.strip()}")
        if character.description.strip():
            printt()
            printt(f"{COL_DIM_ITALICS}{character.description.strip()}")
        printt()


def make_extra_rows(summary: AttributionSummary, roster_ids: set[str]) -> list[tuple[str, int, str]]:
    """
    The (id, count, description) rows for counted speakers that aren't roster
    characters: unknown ids, then the reserved answers. The OTHER row names its
    recurring speakers.
    """
    # Ids neither in the roster nor reserved shouldn't occur (a new roster
    # discards saved ids), but are listed rather than hidden
    others = sorted((label, count, "") for label, count in summary.counts.items()
                    if label not in roster_ids and label not in RESERVED_SPEAKER_IDS)

    descriptions = dict(RESERVED_LABEL_DESCRIPTIONS)
    if summary.recurring_others:
        speakers = ", ".join(f"{name} ({count:,})" for name, count in summary.recurring_others.most_common())
        # Less frequent (and unnamed) OTHER answers are counted but not named
        if summary.counts[OTHER] > sum(summary.recurring_others.values()):
            speakers += ", ..."
        descriptions[OTHER] += f": {speakers}"
    reserved = [(label, summary.counts[label], descriptions[label])
                for label in RESERVED_SPEAKER_ORDER if summary.counts[label]]
    return [*others, *reserved]


def print_extra_rows(
        rows: list[tuple[str, int, str]], first_width: int, id_width: int, tier_gender_gap: bool = False,
) -> None:
    """Prints the non-roster rows; with `tier_gender_gap`, descriptions line up with the table's name column."""
    gap = " " * (TIER_WIDTH + GENDER_WIDTH + 4) if tier_gender_gap else ""
    for label, count, description in rows:
        printt(f"{count:>{first_width},}  {COL_ACCENT}{label:<{id_width}}{COL_DEFAULT}  {gap}{COL_DIM}{description}")


RESERVED_LABEL_DESCRIPTIONS = {
    OTHER: "Not found in roster",
    UNKNOWN: "Speaker couldn't be determined",
    NOT_SPEECH: "Quoted text that isn't speech, like a sign or title",
}
