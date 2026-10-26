"""Step 3: choosing a project voice for each roster character (see plans/llm-speaker-attribution.md)."""
from collections import Counter
from collections.abc import Callable

from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice.voice_menu_shared import (
    VOICES_PER_PAGE,
    VoiceMenuShared,
    make_more_voices_label,
    make_page_breadcrumb,
    make_page_range_text,
)
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.attribution_summary import AttributionSummary
from tts_audiobook_tool.speaker_assign.roster import RosterCharacter
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.voice_apply import VoiceApplyPlan
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import make_hotkey_string, make_noun, print_feedback, printt


def make_step_label(state: State) -> str:
    """The parent menu's label for step 3."""
    label = "Step 3: Map to voices"
    if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
        label += f" {COL_DIM}(requires step 1)"
    return label


def voices_menu(state: State) -> None:

    project_dir = state.project.dir_path
    if not SpeakerAssignFileUtil.has_roster(project_dir):
        print_feedback("Requires step 1 (generate character roster)", is_error=True)
        return
    if len(state.project.voice_references) < 2:
        print_feedback("Requires 2+ voices")
        return
    roster_file = SpeakerAssignFileUtil.load_roster_file(project_dir)
    if isinstance(roster_file, str):
        ask.ask_error(roster_file)
        return
    attributions = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(attributions, str):
        ask.ask_error(attributions)
        return
    loaded_voice_map = SpeakerAssignFileUtil.load_voices_file(project_dir)
    if isinstance(loaded_voice_map, str):
        # Not treated as empty: the next save would overwrite the file
        ask.ask_error(loaded_voice_map)
        return
    # The saved choices, kept current as they're changed here
    voice_map: VoiceMapFile = loaded_voice_map

    def save(new_voice_map: VoiceMapFile) -> None:
        nonlocal voice_map
        if new_voice_map == voice_map:
            return
        error = SpeakerAssignFileUtil.save_voices_file(project_dir, new_voice_map)
        if error:
            ask.ask_error(error)
            return
        voice_map = new_voice_map

    # Counts walk the whole book: computed once on entry (neither the roster
    # nor the attributions can change while in this menu)
    counts = AttributionSummary.make(attributions, state.project.book).counts
    roster_characters = roster_file.roster.characters
    # Before step 2 has run, every count is zero, so the whole roster is listed
    characters = sort_characters(roster_characters, counts, omit_unused=bool(attributions.ranges))

    subheading = VOICES_DESC
    if not roster_characters:
        subheading += f"\n\n{COL_ERROR}The roster has no characters"
    elif not attributions.ranges:
        subheading += f"\n\n{COL_DIM}No dialog has been assigned to characters yet (step 2)."

    def on_character(_: State, item: MenuItem) -> None:
        character: RosterCharacter = item.data
        character_id = character.id
        voice_submenu(
            state,
            heading=f"Assign {character_id} to...",
            current=voice_map.get(character_id),
            clear_label="Clear",
            on_select=lambda index: save(voice_map.with_voice(character_id, index)),
            subheading=make_character_subheading(character),
        )

    def on_set_all(_: State, __: MenuItem) -> None:
        ids = [character.id for character in characters]
        voice_submenu(
            state,
            heading="Set all to...",
            current=None,
            clear_label="Clear all",
            on_select=lambda index: save(VoiceMapFile.all_set_to(ids, index)),
        )

    def make_item(character: RosterCharacter) -> MenuItem:
        return MenuItem(
            make_character_label(state, character.id, counts[character.id], voice_map.get(character.id)),
            on_character,
            data=character,
        )

    def make_action_items() -> list[MenuItem]:
        return [
            MenuItem("Set all to...", on_set_all),
            MenuItem(
                "Apply",
                lambda _, __: run_apply(state, characters, attributions, voice_map),
                blank_line_before=True,
            ),
        ]

    characters_page_menu(state, characters, 0, make_item, make_action_items, subheading)


def characters_page_menu(
        state: State,
        characters: list[RosterCharacter],
        start: int,
        make_item: Callable[[RosterCharacter], MenuItem],
        make_action_items: Callable[[], list[MenuItem]],
        subheading: str,
) -> None:
    """One page of the character list: up to CHARACTERS_PER_PAGE characters,
    then "More characters..." (opens the next page as a submenu) when any are
    left, then the action items, which every page repeats. Paging keeps the
    menu within the hotkeys available; the list is sorted by dialog count, so
    the later pages hold the minor speakers."""
    page = characters[start:start + CHARACTERS_PER_PAGE]
    end = start + len(page)
    remaining = len(characters) - end

    def items_maker(_: State) -> list[MenuItem]:
        items = [make_item(character) for character in page]
        if remaining:
            items.append(MenuItem(
                f"More characters... {COL_DIM}({remaining:,})",
                lambda _, __: characters_page_menu(state, characters, end, make_item, make_action_items, subheading),
            ))
        return items + make_action_items()

    if start == 0:
        heading = breadcrumb = "Map to voices"
    else:
        # The description is only shown on the first page
        subheading = ""
        breadcrumb = f"Characters {start + 1:,}-{end:,}"
        heading = f"Map to voices {COL_DIM}({start + 1:,}-{end:,} of {len(characters):,})"

    MenuUtil.menu(state, heading=heading, items=items_maker, subheading=subheading, breadcrumb=breadcrumb)


def run_apply(
        state: State, characters: list[RosterCharacter], attributions: AttributionsFile, voice_map: VoiceMapFile,
) -> None:
    """Previews the voices to apply, asks to confirm, then sets them on the
    dialog lines and saves the project text."""
    project = state.project
    book = project.book
    roster_ids = {character.id for character in characters}
    chosen = {character_id: index for character_id, index in voice_map.voices.items() if character_id in roster_ids}
    if not chosen:
        print_feedback("No voices chosen")
        return
    missing = [character_id for character_id, index in chosen.items() if index >= len(project.voice_references)]
    if missing:
        ask.ask_error(f"Chosen voice doesn't exist for: {', '.join(missing)}. Choose another voice first.")
        return
    plan = VoiceApplyPlan.make(book, roster_ids, attributions, voice_map)
    if not plan.applied:
        print_feedback("No dialog has been assigned to the chosen characters (step 2)")
        return
    changed = plan.changed_lines(book)
    if not changed:
        print_feedback("Nothing to change: the dialog lines already have these voices")
        return

    segments = project.sound_segments
    generated = set(segments.sound_segments_map.keys()) if segments is not None else set()
    MenuUtil.print_screen_heading(state, "Apply voices", breadcrumb="Apply")
    for line in make_apply_preview(state, characters, voice_map, plan, changed, generated):
        printt(line)
    printt()
    if not ask.ask_confirm(f"Apply? Press {make_hotkey_string('Y')} to confirm: "):
        return

    previous = plan.apply(book)
    if previous:
        error = ProjectTextIOUtil.save_book(project)
        if error:
            VoiceApplyPlan.restore(book, previous)
            ask.ask_error(f"Couldn't save the project text: {error}")
            return
    print_feedback(f"Updated {len(previous):,} {make_noun('line', 'lines', len(previous))}")


def make_apply_preview(
        state: State,
        characters: list[RosterCharacter],
        voice_map: VoiceMapFile,
        plan: VoiceApplyPlan,
        changed: list[int],
        generated: set[int],
) -> list[str]:
    """The confirmation text: characters grouped by voice (with their dialog
    passage counts), what's left unchanged, and any warnings."""
    voice_references = state.project.voice_references
    by_voice: dict[int, list[str]] = {}
    for character in characters:  # Most passages first
        index = voice_map.get(character.id)
        if index is not None:
            by_voice.setdefault(index, []).append(
                f"{COL_ACCENT}{character.id}{COL_DEFAULT} ({plan.applied[character.id]:,})")
    rows = [(VoiceMenuShared.make_voice_title(voice_references[index], index), ", ".join(names))
            for index, names in sorted(by_voice.items())]
    unchanged_counts = SpeakerLabelUtil.sorted_counts(plan.unchanged)
    if unchanged_counts:
        unchanged = ", ".join(f"{label} ({count:,})" for label, count in unchanged_counts)
        rows.append(("Unchanged", f"{unchanged} {COL_DIM}(no voice chosen)"))
    width = max(len(title) for title, _ in rows)

    lines = ["Dialog passages per character:", ""]
    lines += [f"  {title:<{width}}  {names}" for title, names in rows]
    lines.append("")
    lines.append(f"{len(changed):,} {make_noun('line changes', 'lines change', len(changed))} voice. "
                 f"Voices already set on these lines are replaced.")
    if plan.skipped:
        lines.append(f"{COL_DIM}{plan.skipped:,} {make_noun('passage', 'passages', plan.skipped)} skipped: "
                     f"the project text changed after they were assigned (re-run step 2).")
    with_audio = sum(1 for line in changed if line in generated)
    if with_audio:
        lines.append(f"{COL_DIM}{with_audio:,} of these lines already have generated audio, "
                     f"which keeps its old voice until regenerated.")
    return lines


def voice_submenu(
        state: State, heading: str, current: int | None, clear_label: str,
        on_select: Callable[[int | None], None], subheading: str = "",
) -> None:
    """One item per project voice, then `clear_label`; selecting one calls
    `on_select` with the voice index (None for clear) and returns.
    Paged like the Voices menu: past VOICES_PER_PAGE voices, "More voices..."
    opens the next page as a submenu; every page repeats `clear_label` and
    `subheading`."""
    voice_picker_page(state, heading, current, clear_label, on_select, 0, subheading)


def voice_picker_page(
        state: State, heading: str, current: int | None, clear_label: str,
        on_select: Callable[[int | None], None], start: int, subheading: str = "",
) -> bool:
    """One page of `voice_submenu`. Returns whether a choice was made, so that
    a choice closes every page, while going back returns to the previous one."""
    chosen = False

    def choose(index: int | None) -> bool:
        nonlocal chosen
        on_select(index)
        chosen = True
        return True  # Closes this page

    def make_range() -> tuple[int, int]:
        count = len(state.project.voice_references)
        return min(start + VOICES_PER_PAGE, count), count

    def items_maker(_: State) -> list[MenuItem]:
        end, count = make_range()
        items = []
        for index in range(start, end):
            label = VoiceMenuShared.make_voice_title(state.project.voice_references[index], index)
            if index == current:
                label += f" {COL_ACCENT}(selected)"
            items.append(MenuItem(label, lambda _, item: choose(item.data), data=index))
        if end < count:
            items.append(MenuItem(
                make_more_voices_label(count - end),
                lambda _, __: voice_picker_page(state, heading, current, clear_label, on_select, end, subheading),
            ))
        items.append(MenuItem(clear_label, lambda _, __: choose(None)))
        return items

    if start == 0:
        page_heading = breadcrumb = heading
    else:
        end, count = make_range()
        page_heading = f"{heading} {COL_DIM}{make_page_range_text(start, end, count)}"
        breadcrumb = make_page_breadcrumb(start, end)
    MenuUtil.menu(state, heading=page_heading, items=items_maker, subheading=subheading, breadcrumb=breadcrumb)
    return chosen


def sort_characters(
        characters: tuple[RosterCharacter, ...], counts: Counter[str], omit_unused: bool = False,
) -> list[RosterCharacter]:
    """Most dialog passages first; ties keep roster order. With omit_unused,
    characters with no dialog passages are left out."""
    if omit_unused:
        characters = tuple(character for character in characters if counts[character.id])
    return sorted(characters, key=lambda character: -counts[character.id])


def make_character_subheading(character: RosterCharacter) -> str:
    """The character's user-facing (spoiler-free) roster fields, to help choose
    a voice: eg "Baron de Verdira (male)" then "voice: elderly; deep, gravelly"."""
    details = [part for part in (character.gender, "narrator" if character.is_narrator else "")
               if part and part != "unknown"]
    lines = [f"{COL_DEFAULT}{character.name}" + (f" {COL_DIM}({', '.join(details)})" if details else "")]
    if character.voice_hint.strip():
        lines.append(f"{COL_DIM}voice: {COL_DEFAULT}{character.voice_hint.strip()}")
    return "\n".join(lines)


def make_character_label(state: State, character_id: str, count: int, voice_index: int | None) -> str:
    """Eg "BOB (50) -> Voice 1 bob"; just "BOB (50)" when no voice is chosen."""
    label = f"{COL_DEFAULT}{character_id} {COL_DIM}({count:,})"
    if voice_index is None:
        return label
    label += f" {COL_DEFAULT}->"
    voice_references = state.project.voice_references
    if voice_index >= len(voice_references):
        # Eg the voice was removed after it was chosen
        return label + f" {COL_ERROR}Voice {voice_index + 1} (missing)"
    title = VoiceMenuShared.make_voice_title(voice_references[voice_index], voice_index)
    return label + f" {COL_ACCENT}{title}"


# Characters listed per page: the 35 auto-assigned hotkeys (1-9, a-z) minus
# "More characters...", "Set all to..." and "Apply", so a full page ends on "Z"
CHARACTERS_PER_PAGE = 32

VOICES_DESC = \
"""Sets each character's dialog lines (as identified in step 2) to the voice
chosen for that character.

Choose voices below, then select \"Apply\"."""
