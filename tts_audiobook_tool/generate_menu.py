from __future__ import annotations

from tts_audiobook_tool.app_types import Strictness, SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.concat_menu import ConcatMenu
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class GenerateMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_start_label(_: State) -> str:
            label = "Start"
            err = AppUtil.get_combined_prereq_error(state.project, short_format=True)
            if err:
                return make_menu_label(label, err, value_prefix="", color_code=COL_ERROR)
            else:
                return label

        def make_range_label(_: State) -> str:
            if not state.project.generate_range_string:
                range_label = f"{COL_DIM}(currently set to: {COL_ACCENT}all{COL_DIM})"
            else:
                line_word = make_noun("line", "lines", len(state.project.get_indices_to_generate()))
                range_label = f"{COL_DIM}(currently set to: {COL_ACCENT}{line_word} {state.project.generate_range_string}{COL_DIM})"

            if not state.project.generate_range_string:
                complete_label = ""
            else:
                selected_indices = state.project.get_indices_to_generate()
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
            qualifier = " in specified range" if state.project.generate_range_string else ""
            regenerate_label = f"{COL_DIM}(currently: {COL_ACCENT}{failed_items_label}{COL_DIM}{qualifier})"
            return f"Regenerate segments tagged with potential errors {regenerate_label}"

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
            return f"Generate audio {num_complete_label}"

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
                MenuItem(make_regen_label, lambda _, __: do_generate(state, is_regen=True))
            )
            # Batch size
            if Tts.get_type().value.can_batch:
                items.append(
                    MenuItem(make_batch_size_label, lambda _, __: ask_batch_size(state))
                )
            elif Tts.get_type() == TtsModelInfos.VIBEVOICE:
                items.append(
                    MenuItem(make_batch_size_label, lambda _, __: ask_batch_size(state))
                )
            # Strictness
            items.append(MenuItem(make_strictness_label, lambda _, __: GenerateMenu.strictness_menu(state)))
            # Num retries
            items.append(MenuItem(make_retries_label, lambda _, __: ask_retries(state)))
            
            if state.project.sound_segments.num_generated() > 0:
                items.append(MenuItem(f"Delete segments", lambda _, __: ask_delete_segments(state)))

            return items
        
        MenuUtil.menu(state, heading_maker, items_maker)

    @staticmethod
    def strictness_menu(state: State) -> None:

        def on_select(value: Strictness) -> None:
            state.project.strictness = value
            state.project.save()
            print_feedback(f"Strictness set to:", state.project.strictness.label)

        warning_high = Tts.get_class().get_strictness_warning(Strictness.HIGH, state.project, Tts.get_instance_if_exists())

        if state.project.language_code != "en":
            low_desc = f"{Ansi.ITALICS}Highly recommended{Ansi.RESET}{COL_DIM} for current language code {state.project.language_code}"
            medium_desc = ""
            high_desc = ""
        else:
            low_desc = ""
            medium_desc = ""
            high_desc = warning_high if warning_high else "Best net accuracy but triggers more retries"

        MenuUtil.options_menu(
            state=state,
            heading_text=make_strictness_label(state),
            labels=[item.label for item in list(Strictness)],
            sublabels=[low_desc, medium_desc, high_desc],
            values=[item for item in list(Strictness)],
            current_value=state.project.strictness,
            default_value=None,
            on_select=on_select,
            subheading=STRICTNESS_DESC
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
    
    print_heading("Delete segments")

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
        s = f"All generated segments will be deleted"
    else:
        segment_word = make_noun("segment", "segments", len(indices_to_delete))
        s = f"The following {len(indices_to_delete)} {segment_word} will be deleted: \n"
        s += ParseUtil.make_ranges_string(selected_indices, total_num_items)
    printt(s)
    if not AskUtil.ask_confirm():
        return
    
    state.project.sound_segments.delete_by_indices(indices_to_delete)

def make_strictness_label(state: State) -> str:
    label = make_menu_label(
        label="Transcript validation strictness", 
        value=state.project.strictness.label
    )
    warning = Tts.get_class().get_strictness_warning(state.project.strictness, state.project, Tts.get_instance_if_exists())
    if warning:
        label += f"{COL_ERROR}*"
    return label

def make_retries_label(state: State) -> str:
    return make_menu_label(
        label="Transcript validation max retries", 
        value=state.project.max_retries, 
        default=PROJECT_MAX_RETRIES_DEFAULT
    )

def ask_retries(state: State) -> None:
    print_heading(make_retries_label(state))
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

def do_generate(state: State, is_regen: bool) -> None:

    # Get indices to generate, and check if already generated
    if is_regen:
        indices = state.project.sound_segments.get_failed_indices_in_generate_range()
    else:
        indices = state.project.get_selected_indices_not_generated()
    
    if not indices:
        qualifier = " in currently selected range" if state.project.generate_range_string else ""
        if is_regen:
            message = f"No failed items to regenerate{qualifier}."
        else:
            message = f"All items{qualifier} already generated."            
        print_feedback(message)
        return

    # Check model and other app prereqs
    error = AppUtil.get_combined_prereq_error(state.project, short_format=False)
    if error:
        print_feedback(error, is_error=True)
        return

    # Show pre-inference hint/warning if necessary
    if is_regen:
        Hint.show_hint_if_necessary(state.prefs, HINT_REGEN)
    else:
        AppUtil.show_pre_inference_hints(state.prefs, state.project)

    # Print confirmation info, and confirm
    s = f"Will generate {len(indices)} lines in range {state.project.generate_range_string}"
    if not is_regen:
        num = state.project.sound_segments.num_generated_in_current_range()
        if num:
            s += f" {COL_DIM}({num} already complete)"
    printt(s)
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
    print_heading(message, dont_clear=True)
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

    if not is_regen:
        s = f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('C')} to concatenate files now: \a"
        hotkey = AskUtil.ask_hotkey(s)
        printt() # TODO revisit
        if hotkey == "c":
            ConcatMenu.menu(state)

STRICTNESS_DESC = \
"""Dictates how \"strict\" is the transcript validation.
A low strictness value allows for more word errors before 
an audio generation is marked as failed, triggering a retry.
"""

RETRIES_DESC = \
"""This is the max number of retries an audio generation will be attempted 
when speech-to-text validation fails due to too many word errors.
Higher values have diminishing returns.
"""
