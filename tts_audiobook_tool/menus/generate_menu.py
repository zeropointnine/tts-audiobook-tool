from __future__ import annotations

from tts_audiobook_tool.app_types import Strictness, SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.prereqs_util import PrereqUtil
from tts_audiobook_tool.project_util import ProjectUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whitelist import Whitelist

class GenerateMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_start_label(_: State) -> str:
            prereq = PrereqUtil.get_generate_prereq_error_string(state, verbose=False)
            label = "Start"
            if prereq:
                label += f" {COL_DIM}({COL_ERROR}{prereq}{COL_DIM})"
            return label

        def make_range_label(_: State) -> str:
            if not state.project.generate_range_string:
                range_label = f"{COL_DIM}(currently set to: {COL_ACCENT}all{COL_DIM})"
            else:
                line_word = make_noun("line", "lines", len(ProjectUtil.get_indices_to_generate(state.project)))
                range_label = f"{COL_DIM}(currently set to: {COL_ACCENT}{line_word} {state.project.generate_range_string}{COL_DIM})"

            if not state.project.generate_range_string:
                complete_label = ""
            else:
                selected_indices = ProjectUtil.get_indices_to_generate(state.project)
                all_generated_indices = state.project.sound_segments.sound_segments_map.keys()
                selected_indices_not_generated = selected_indices - all_generated_indices
                num_selected_indices_generated = len(selected_indices) - len(selected_indices_not_generated)
                complete_label = f"({COL_ACCENT}{num_selected_indices_generated}{COL_DIM} of {COL_ACCENT}{len(selected_indices)}{COL_DIM} complete)"

            if not state.project.generate_range_string:
                label = "Specify line range"
            else:
                label = "Line range"
            return f"{label} {range_label} {complete_label}"

        def make_regen_label(_: State) -> str:
            num_fails = len( state.project.sound_segments.get_failed_indices_in_generate_range() )
            failed_items_label = f"{num_fails} {make_noun('item', 'items', num_fails)}"
            currently = f"{COL_DIM}(currently: {COL_ACCENT}{failed_items_label}{COL_DIM})"
            return f"Regenerate segments with errors {currently}"

        def make_batch_size_label(state: State) -> str:
            value = state.project.batch_size
            if value == -1:
                value = 1
            value_string = "disabled" if value == 1 else str(value)
            s = "Batch size "
            currently = make_currently_string(value_string)
            s = s + currently
            return s

        # Menu
        def heading_maker(_: State) -> str:
            total_segments_generated = state.project.sound_segments.num_generated()
            num_complete_label = f"{COL_DIM}({COL_ACCENT}{total_segments_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.phrase_groups)}{COL_DIM} total lines complete)"
            return f"Generate audio segments {num_complete_label}"

        def items_maker(_: State) -> list[MenuItem]:
            items = []
            # Start
            items.append(
                MenuItem(make_start_label, lambda _, __: do_generate(state, is_regen=False)),
            )
            # Range
            items.append(
                MenuItem(make_range_label, lambda _, __: ask_item_range(state)),
            )
            # Re-generate
            items.append(
                MenuItem(
                    make_regen_label, 
                    lambda _, __: regenerate_menu(state)
                )
            )
            if state.project.sound_segments.num_generated() > 0:
                items.append(MenuItem(f"Delete segments", lambda _, __: ask_delete_segments(state)))
            
            # Batch size
            if Tts.get_type().value.can_batch:
                items.append(
                    MenuItem(make_batch_size_label, lambda _, __: ask_batch_size(state), superlabel="Options")
                )
            # Tolerance
            items.append(
                MenuItem(
                    make_tolerance_label, lambda _, __: GenerateMenu.tolerance_menu(state),
                    superlabel="Options"
                )
            )
            
            # Max retries
            items.append(
                MenuItem(
                    make_retries_label, lambda _, __: ask_retries(state), 
                )
            )

            # Limit silence gaps
            items.append(
                MenuItem(
                    make_limit_silence_gaps_label, lambda _, __: GenerateMenu.limit_silence_gaps_menu(state),
                    superlabel="Post-processing"
                )
            )

            return items
        
        MenuUtil.menu(state, heading_maker, items_maker, breadcrumb="Generate")

    @staticmethod
    def limit_silence_gaps_menu(state: State) -> None:

        def make_enabled_label(_: State) -> str:
            value = state.project.limit_silence_gaps
            value_str = "True" if value else "False"
            label = f"Enabled: {COL_ACCENT}{value_str}"
            if value == PROJECT_DEFAULT_LIMIT_SILENCE_GAPS:
                label += f" {COL_DIM}(default)"
            return label

        def items_maker(_: State) -> list[MenuItem]:
            items = []
            # Enabled
            items.append(
                MenuItem(
                    make_enabled_label,
                    lambda _, __: GenerateMenu._limit_silence_gaps_enabled_menu(state),
                )
            )
            # Gap duration threshold
            items.append(
                MenuUtil.make_number_item(
                    state=state,
                    attr="limit_silence_gaps_duration",
                    base_label="Gap duration threshold",
                    default_value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
                    is_minus_one_default=False,
                    num_decimals=2,
                    prompt="Enter gap duration threshold (seconds):",
                    min_value=SILENCE_GAP_DURATION_MIN,
                    max_value=SILENCE_GAP_DURATION_MAX,
                )
            )
            return items

        MenuUtil.menu(
            state=state,
            heading="Limit silence gaps",
            items=items_maker,
            subheading=LIMIT_SILENCE_GAPS_MENU_SUBHEADING,
            breadcrumb="Limit silence gaps",
        )

    @staticmethod
    def _limit_silence_gaps_enabled_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.limit_silence_gaps = value
            state.project.save()
            print_feedback(f"Limit silence gaps set to: {value}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Enabled",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.limit_silence_gaps,
            default_value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS,
            on_select=on_select,
            breadcrumb="Limit silence gaps > Enabled",
        )

    @staticmethod
    def tolerance_menu(state: State) -> None:

        def on_select(value: Strictness) -> None:
            state.project.strictness = value
            state.project.save()
            print_feedback(f"Word error tolerance set to:", state.project.strictness.label)

        warning_high = Tts.get_class().get_strictness_warning(Strictness.HIGH, state.project, Tts.get_instance_if_exists())

        if not Whitelist.supports_language(state.project.language_code):
            low_desc = f"{Ansi.ITALICS}Highly recommended when language is not {list(Whitelist.LANGUAGES.keys())}"
            medium_desc = ""
            high_desc = ""
            intolerant_desc = ""
        else:
            low_desc = "Allows more word errors. Segments pass unless notably off."
            medium_desc = "Balanced. Reasonable choice for most languages."
            high_desc = warning_high if warning_high else "Strict; segments with minor word errors will be flagged for regeneration."
            intolerant_desc = "Zero tolerance. Segments with even one word error are flagged for regeneration.\n      Best net accuracy; for the time and compute unconstrained only."

        MenuUtil.options_menu(
            state=state,
            heading_text=make_tolerance_label(state),
            labels=[item.label for item in list(Strictness)],
            sublabels=[low_desc, medium_desc, high_desc, intolerant_desc],
            values=[item for item in list(Strictness)],
            current_value=state.project.strictness,
            default_value=None,
            on_select=on_select,
            subheading=STRICTNESS_DESC,
            breadcrumb="Word error tolerance",
        )

# ---

def ask_item_range(state: State) -> None:

    num_items = len(state.project.phrase_groups)

    s = state.project.generate_range_string if state.project.generate_range_string else "all"
    printt(f"Enter line numbers to generate {COL_DIM}(currently: {s}):") 
    printt(f"{COL_DIM}For example, \"1-100\" or \"201-210, 215\", or just \"all\"") 

    inp = AskUtil.ask()
    if inp == "all" or inp == "a":
        indices = set( [item for item in range(0, num_items)] )
    else:
        indices, warnings = ParseUtil.parse_ranges_string(inp, num_items)
        if not indices:
            return
        if warnings:
            print_feedback("\n".join(warnings))
            return

    s = ParseUtil.make_ranges_string(indices, len(state.project.phrase_groups))
    state.project.generate_range_string = "" if s == "all" else s
    state.project.save()

    print_feedback(f"Range set to: {s}")

def ask_delete_segments(state: State) -> None:

    if state.project.sound_segments.num_generated() == 0:
        print_feedback("Nothing to delete")
        return
    
    MenuUtil.print_screen_heading(state, "Delete segments", breadcrumb="Delete segments")

    path = os.path.join(state.project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
    hint = Hint.make_using(HINT_DELETE_SEGMENTS, make_terminal_hyperlink(path, is_file=True))
    Hint.show_hint_if_necessary(state.prefs, hint)

    printt(f"Enter line numbers to delete:") 
    printt(f"{COL_DIM}For example, \"3, 5, 21\" or \"201-350, 215\", or just \"all\"") 
    inp = AskUtil.ask()
    if not inp:
        return

    # Make selected indices from input
    total_num_items = len(state.project.phrase_groups)
    if inp == "all" or inp == "a":
        selected_indices = set( [item for item in range(0, total_num_items)] )
    else:
        selected_indices, _ = ParseUtil.parse_ranges_string(inp, total_num_items)
        if not selected_indices:
            print_feedback("No valid line numbers entered")
            return
        
    existing_indices = state.project.sound_segments.get_existing_indices()
    indices_to_delete = selected_indices & existing_indices
    if not indices_to_delete:
        print_feedback("No valid line numbers entered")
        return

    if len(indices_to_delete) == len(existing_indices):
        noun = make_noun("segment", "segments", len(indices_to_delete))
        s = f"All {len(indices_to_delete)} generated {noun} will be deleted"
    else:
        segment_word = make_noun("segment", "segments", len(indices_to_delete))
        s = f"The following {len(indices_to_delete)} {segment_word} will be deleted: \n"
        s += ParseUtil.make_ranges_string(indices_to_delete, total_num_items)
    printt(s)
    if not AskUtil.ask_confirm():
        return
    
    state.project.sound_segments.delete_by_indices(indices_to_delete)

    # NB, without this delay, directory watcher may not update in time for next menu printout
    print_feedback(f"Deleted {len(indices_to_delete)} segments")

def make_tolerance_label(state: State) -> str:
    label = make_menu_label(
        label="Word error tolerance",
        value=state.project.strictness.label
    )
    warning = Tts.get_class().get_strictness_warning(state.project.strictness, state.project, Tts.get_instance_if_exists())
    if warning:
        label += f"{COL_ERROR}*"
    return label

def make_retries_label(state: State) -> str:
    return make_menu_label(
        label="Generation max retries",
        value=state.project.max_retries, 
        default=PROJECT_MAX_RETRIES_DEFAULT
    )

def make_limit_silence_gaps_label(state: State) -> str:
    return make_menu_label(
        label="Limit silence gaps",
        value=state.project.limit_silence_gaps,
        default=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS
    )

def ask_retries(state: State) -> None:
    MenuUtil.print_heading(state, make_retries_label(state))
    printt(RETRIES_DESC)
    AskUtil.ask_number(
        state.project,
        "max_retries",
        "Enter value:",
        PROJECT_MAX_RETRIES_MIN, PROJECT_MAX_RETRIES_MAX, PROJECT_MAX_RETRIES_DEFAULT, "Max retries set to:", is_int=True
    )

def ask_batch_size(state: State) -> None:

    field_name = Tts.get_type().value.batch_size_project_field
    if not field_name:
        return # silently ignore (shouldn't happen)

    Hint.show_hint_if_necessary(state.prefs, HINT_BATCH)

    prompt = "Enter batch size:"
    AskUtil.ask_number(
        state.project, field_name, prompt,
        1, PROJECT_BATCH_SIZE_MAX,
        PROJECT_BATCH_SIZE_DEFAULT, "Set batch size:", is_int=True
    )

def make_regenerate_segments_with_errors_desc(state: State) -> str:
    return REGENERATE_SEGMENTS_WITH_ERRORS_DESC.replace("%1", state.project.strictness.label)

def regenerate_menu(state: State) -> None:

    def make_print_label(_: State) -> str:
        num_fails = len(state.project.sound_segments.get_failed_indices_in_generate_range())
        return f"Review lines to be regenerated {COL_DIM}({COL_ACCENT}{num_fails}{COL_DIM})"

    def on_start(_: State, __: MenuItem) -> None:
        do_generate(state, is_regen=True, show_stt_status=False)

    def on_print(_: State, __: MenuItem) -> None:
        indices = state.project.sound_segments.get_failed_indices_in_generate_range()
        AppUtil.print_regen_lines(state.project, indices)
        AskUtil.ask_enter_to_continue()

    items = [
        MenuItem("Start", on_start),
        MenuItem(make_print_label, on_print),
    ]

    MenuUtil.menu(
        state,
        "Regenerate segments with errors",
        items,
        subheading=make_regenerate_segments_with_errors_desc,
        breadcrumb="Regenerate segments with errors",
    )

def do_generate(state: State, is_regen: bool, show_stt_status: bool = True) -> None:

    # Check prereqs
    error = PrereqUtil.get_generate_prereq_error_string(state, verbose=True)
    if error:
        print_feedback(error, is_error=True)
        return

    # Get indices to generate, and check if already generated
    if is_regen:
        indices = state.project.sound_segments.get_failed_indices_in_generate_range()
    else:
        indices = ProjectUtil.get_selected_indices_not_generated(state.project)
    
    if not indices:
        qualifier = " in currently selected range" if state.project.generate_range_string else ""
        if is_regen:
            message = f"No segments with errors to regenerate{qualifier}."
        else:
            message = f"All items{qualifier} already generated."            
        print_feedback(message)
        return

    # Show pre-inference hint/warning if necessary
    if is_regen:
        Hint.show_hint_if_necessary(state.prefs, HINT_REGEN)
    else:
        AppUtil.show_pre_inference_hints(state.prefs, state.project)

    # Print confirmation info, and confirm
    if not is_regen:
        s = f"Will generate {len(indices)} lines in range {state.project.generate_range_string}"
        num = state.project.sound_segments.num_generated_in_current_range()
        if num:
            s += f" {COL_DIM}({num} already complete)"
        printt(s)
        if show_stt_status:
            if not Stt.should_skip(state):
                s = "Speech-to-text validation enabled"
                s += f" {COL_DIM}({Stt.short_description()})"
            else:
                s = "Speech-to-text validation disabled"
            printt(s)
            printt()
        b = AskUtil.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
        if not b:
            return

    # Print heading
    word = "Regenerating" if is_regen else "Generating"
    message = f"{word} {len(indices)} audio segment/s..."
    if state.prefs.stt_variant == SttVariant.DISABLED:
        message += f" {COL_DIM}(speech-to-text validation disabled){COL_DEFAULT}"
    MenuUtil.print_heading(state, message, dont_clear=True)
    printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
    printt()

    # Generate
    did_interrupt = GenerateUtil.generate_files(
        state=state,
        indices_set=indices,
        batch_size=state.project.batch_size,
        is_regen=is_regen
    )

    if did_interrupt:
        AskUtil.ask_enter_to_continue()
        return

    AppUtil.play_done_sound()

    if is_regen:
        AskUtil.ask_enter_to_continue()
        return
    
    s = f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('C')} to create audiobook file now: \a"
    hotkey = AskUtil.ask_hotkey(s)
    printt() # TODO revisit
    if hotkey == "c":
        ConcatMenu.menu(state)

STRICTNESS_DESC = \
"""Controls how many word errors are acceptable per segment.
Applies during generation (auto-retry) and when identifying
existing segments for regeneration.
"""

RETRIES_DESC = \
"""This is the max number of retries an audio generation will be attempted 
when speech-to-text validation fails due to too many word errors.
Higher values have diminishing returns.
"""

REGENERATE_SEGMENTS_WITH_ERRORS_DESC = \
"""Retry only the segments currently flagged as having word errors.
This uses your current line range and word error tolerance setting (%1)
to decide which segments should be regenerated.
"""

LIMIT_SILENCE_GAPS_MENU_SUBHEADING = \
"""Prevents instances of silence within each sound segment from extending beyond
the given duration.

Larger values can be used to prevent long pauses (eg, 1-2 seconds). 

Small values can be used to influence pacing and prosody (eg, 0.0-0.3 seconds).
Best used with \"Text segmentation strategy: Normal.\" 

This setting also applies to realtime playback, voice chat, stand-alone server.
"""
