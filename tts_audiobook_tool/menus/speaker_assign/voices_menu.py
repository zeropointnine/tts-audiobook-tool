"""Step 3: choosing a project voice for each roster character (see plans/llm-speaker-attribution.md)."""
from collections import Counter
from collections.abc import Callable

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.attribution_summary import AttributionSummary
from tts_audiobook_tool.speaker_assign.roster import RosterCharacter
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.voice_apply import VoiceApplyPlan
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import make_hotkey_string, make_noun, print_feedback, printt


def make_step_label(state: State) -> str:
    """The parent menu's label for step 3."""
    label = "Step 3: Assign voices to characters"
    if not SpeakerAssignFileUtil.has_roster(state.project.dir_path):
        label += f" {COL_DIM}(requires step 1)"
    return label


def voices_menu(state: State) -> None:

    project_dir = state.project.dir_path
    if not SpeakerAssignFileUtil.has_roster(project_dir):
        print_feedback("Requires step 1 (create character roster)", is_error=True)
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
    characters = sort_characters(roster_file.roster.characters, counts)

    subheading = VOICES_DESC
    if not characters:
        subheading += f"\n\n{COL_ERROR}The roster has no characters"
    elif not attributions.ranges:
        subheading += f"\n\n{COL_DIM}No dialog has been assigned to characters yet (step 2)."

    def on_character(_: State, item: MenuItem) -> None:
        character_id: str = item.data
        voice_submenu(
            state,
            heading=f"Assign {character_id} to...",
            current=voice_map.get(character_id),
            clear_label="Clear",
            on_select=lambda index: save(voice_map.with_voice(character_id, index)),
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

    def items_maker(_: State) -> list[MenuItem]:
        items = [
            MenuItem(
                make_character_label(state, character.id, counts[character.id], voice_map.get(character.id)),
                on_character,
                data=character.id,
            )
            for character in characters
        ]
        items.append(MenuItem("Set all to...", on_set_all))
        items.append(MenuItem(
            "Apply",
            lambda _, __: run_apply(state, characters, attributions, voice_map),
            blank_line_before=True,
        ))
        return items

    MenuUtil.menu(
        state,
        heading="Assign voices to characters",
        items=items_maker,
        subheading=subheading,
        breadcrumb="Assign voices to characters",
    )


def run_apply(
        state: State, characters: list[RosterCharacter], attributions: AttributionsFile, voice_map: VoiceMapFile,
) -> None:
    """Previews the voices to apply, asks to confirm, then sets them on the
    dialog lines and saves the project text. Also switches the project to
    user-defined voice selection, without which line voices aren't used."""
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
    switch_mode = project.voice_select_mode is not VoiceSelectMode.USER_DEFINED
    if not changed and not switch_mode:
        print_feedback("Nothing to change: the dialog lines already have these voices")
        return

    segments = project.sound_segments
    generated = set(segments.sound_segments_map.keys()) if segments is not None else set()
    MenuUtil.print_screen_heading(state, "Apply voices", breadcrumb="Apply")
    for line in make_apply_preview(state, characters, voice_map, plan, changed, generated, switch_mode):
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
    feedback = f"Updated {len(previous):,} {make_noun('line', 'lines', len(previous))}"
    if switch_mode:
        project.voice_select_mode = VoiceSelectMode.USER_DEFINED
        error = project.save()
        if error:
            ask.ask_error(f"{feedback}, but couldn't save the voice selection mode: {error}")
            return
        feedback += "; voice selection mode set to user-defined"
    print_feedback(feedback)


def make_apply_preview(
        state: State,
        characters: list[RosterCharacter],
        voice_map: VoiceMapFile,
        plan: VoiceApplyPlan,
        changed: list[int],
        generated: set[int],
        switch_mode: bool,
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
    if plan.unchanged:
        unchanged = ", ".join(f"{label} ({count:,})" for label, count in plan.unchanged.most_common())
        rows.append(("Unchanged", f"{unchanged} {COL_DIM}(no voice chosen)"))
    width = max(len(title) for title, _ in rows)

    lines = ["Dialog passages per character:", ""]
    lines += [f"  {title:<{width}}  {names}" for title, names in rows]
    lines.append("")
    if changed:
        lines.append(f"{len(changed):,} {make_noun('line changes', 'lines change', len(changed))} voice. "
                     f"Voices already set on these lines are replaced.")
    else:
        lines.append("The dialog lines already have these voices.")
    if 0 in by_voice:
        lines.append(f"{COL_ERROR}Note: dialog set to Voice 1 looks like narration afterwards, "
                     f"so step 2 won't see it again.")
    if plan.skipped:
        lines.append(f"{COL_DIM}{plan.skipped:,} {make_noun('passage', 'passages', plan.skipped)} skipped: "
                     f"the project text changed after they were assigned (re-run step 2).")
    with_audio = sum(1 for line in changed if line in generated)
    if with_audio:
        lines.append(f"{COL_DIM}{with_audio:,} of these lines already have generated audio, "
                     f"which keeps its old voice until regenerated.")
    if switch_mode:
        mode = state.project.voice_select_mode
        lines.append(f"Voice selection mode will be set to {COL_ACCENT}user-defined{COL_DEFAULT} "
                     f"(currently: {mode.current_label}); line voices are only used in that mode.")
    return lines


def voice_submenu(
        state: State, heading: str, current: int | None, clear_label: str,
        on_select: Callable[[int | None], None],
) -> None:
    """One item per project voice, then `clear_label`; selecting one calls
    `on_select` with the voice index (None for clear) and returns."""
    items = []
    for index, voice in enumerate(state.project.voice_references):
        label = VoiceMenuShared.make_voice_title(voice, index)
        if index == current:
            label += f" {COL_ACCENT}(selected)"
        items.append(MenuItem(label, lambda _, item: on_select(item.data), data=index))
    items.append(MenuItem(clear_label, lambda _, __: on_select(None)))
    MenuUtil.menu(state, heading=heading, items=items, one_shot=True, breadcrumb=heading)


def sort_characters(characters: tuple[RosterCharacter, ...], counts: Counter[str]) -> list[RosterCharacter]:
    """Most dialog passages first; ties keep roster order."""
    return sorted(characters, key=lambda character: -counts[character.id])


def make_character_label(state: State, character_id: str, count: int, voice_index: int | None) -> str:
    """Eg "BOB (50) -> Voice 1 bob"; just "BOB (50)" when no voice is chosen."""
    label = f"{COL_ACCENT}{character_id} {COL_DEFAULT}({count:,})"
    if voice_index is None:
        return label
    voice_references = state.project.voice_references
    if voice_index >= len(voice_references):
        # Eg the voice was removed after it was chosen
        return label + f" -> {COL_ERROR}Voice {voice_index + 1} (missing)"
    title = VoiceMenuShared.make_voice_title(voice_references[voice_index], voice_index)
    return label + f" -> {COL_ACCENT}{title}"


VOICES_DESC = \
"""Choose a project voice for each character in the roster. The number after each
character is how many dialog passages were assigned to them in step 2. Apply sets
the voice of each character's dialog lines in the project text. Narration, and the
dialog of characters without a chosen voice, is not changed."""
