from __future__ import annotations

from typing import Iterable

from tts_audiobook_tool.app_types import Strictness, VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.app_support import app_hint_util, hints
from tts_audiobook_tool import ask
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool import readiness
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    EditorClosed,
    EditorSaveFailed,
    run_content_textual_app,
)
from tts_audiobook_tool.textual.generate_editor import (
    GenerateEditor,
    QuickGenerationRequested,
)
from tts_audiobook_tool.textual.generation_app import run_generation_modal
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *
from tts_audiobook_tool.text_ops.whitelist import Whitelist

class GenerateMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_start_label(_: State) -> str:

            label = "Start"

            ungenerated_indices = {
                phrase_index
                for phrase_index in range(len(state.project.phrase_groups))
                if state.project.sound_segments.get_best_item_for(phrase_index) is None
            }
            queued_count = len(
                ProjectUtil.get_indices_to_generate(state.project)
                & ungenerated_indices
            )
            if ungenerated_indices and queued_count == len(ungenerated_indices):
                generated_count = state.project.sound_segments.num_generated()
                if generated_count:
                    remaining_count = len(ungenerated_indices)
                    noun = make_noun("segment", "segments", remaining_count)
                    queued_label = (
                        f"all {remaining_count} remaining {noun} queued"
                    )
                else:
                    queued_label = "all"
            else:
                noun = make_noun("line", "lines", queued_count)
                queued_label = f"{queued_count} {noun} queued"
            label += f" {COL_DIM}({queued_label})"

            blocker = readiness.get_generate_blocker_text(state, verbose=False)
            if blocker:
                label += f" {COL_DIM}({COL_ERROR}{blocker}{COL_DIM})"

            return label

        def make_batch_size_label(state: State) -> str:
            value = ProjectVoiceUtil.get_batch_size(state.project)
            value_string = "disabled" if value == 1 else str(value)
            s = "Concurrent requests " if Tts.is_sgl_mode() else "Batch size "
            currently = make_currently_string(value_string)
            s = s + currently
            return s

        # Menu
        def heading_maker(_: State) -> str:
            return "Generate sound segments"

        def items_maker(_: State) -> list[MenuItem]:

            items = []
            # Generate
            items.append(MenuItem(make_start_label,lambda _, __: do_generate(state)))

            items.append(
                MenuItem(
                    "Select lines / review sound segments",
                    lambda _, __: GenerateMenu.run_editor(state),
            ))

            show_batch_item = Tts.get_type().value.can_batch

            # Batch size
            if show_batch_item:
                items.append(
                    MenuItem(make_batch_size_label, lambda _, __: ask_batch_size(state), superlabel="Options")
                )

            # Tolerance
            items.append(
                MenuItem(
                    make_tolerance_label, lambda _, __: GenerateMenu.tolerance_menu(state),
                    superlabel="Options" if not show_batch_item else ""
                )
            )

            # Max retries
            items.append(
                MenuItem(
                    make_retries_label, lambda _, __: ask_retries(state),
                )
            )

            items.append(
                MenuItem(
                    make_gen_auto_concat_label, lambda _, __: GenerateMenu.gen_auto_concat_menu(state),
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
    def run_editor(state: State) -> None:
        """ Run generate editor """

        range_save_error = ProjectUtil.persist_range_without_generated_items(
            state.project
        )
        if range_save_error:
            print_feedback(range_save_error, is_error=True)

        # Loop is required to re-run editor after "quick-gen"
        quick_gen_index: int | None = None
        while True:
            run_result = run_content_textual_app(
                GenerateEditor(
                    state,
                    quick_gen_index=quick_gen_index,
                )
            )
            if not isinstance(run_result, ContentAppCompleted):
                print_feedback(run_result.message, is_error=True)
                return

            editor_result = run_result.result
            if isinstance(editor_result, EditorSaveFailed):
                print_feedback(editor_result.error, is_error=True)
                return
            if isinstance(editor_result, QuickGenerationRequested):
                if editor_result.save_error:
                    print_feedback(editor_result.save_error, is_error=True)
                quick_gen_index = editor_result.phrase_index
                GenerateUtil.do_quick_generate(
                    state, quick_gen_index
                )
                continue
            if isinstance(editor_result, EditorClosed):
                return

    @staticmethod
    def limit_silence_gaps_menu(state: State) -> None:

        def make_enabled_label(_: State) -> str:
            value = state.project.limit_silence_gaps
            value_str = "True" if value else "False"
            label = f"Enabled: {COL_ACCENT}{value_str}"
            if value == PROJECT_DEFAULT_LIMIT_SILENCE_GAPS:
                label += f" {COL_DIM}(default)"
            return label

        def make_items(_: State) -> list[MenuItem]:
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
            items=make_items,
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
            heading_text="Limit silence gaps",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.limit_silence_gaps,
            default_value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS,
            on_select=on_select,
            breadcrumb="Limit silence gaps > Enabled",
        )

    @staticmethod
    def gen_auto_concat_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.gen_auto_concat = value
            state.project.save()
            print_feedback(f"Concatenate when finished set to: {value}")

        SUBHEADING = 'Automatically runs concatenation step ("Create audiobook file")\nwhen job is finished.\n'

        MenuUtil.options_menu(
            state=state,
            heading_text='Concatenate when finished',
            subheading=SUBHEADING,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.gen_auto_concat,
            default_value=PROJECT_DEFAULT_GEN_AUTO_CONCAT,
            on_select=on_select,
            breadcrumb="Concatenate when finished",
        )

    @staticmethod
    def tolerance_menu(state: State) -> None:

        def on_select(value: Strictness) -> None:
            state.project.strictness = value
            state.project.save()
            print_feedback(f"Word error tolerance set to:", state.project.strictness.label)

        warning_high = Tts.get_class().get_strictness_warning(Strictness.HIGH, state.project, None)

        if not Whitelist.supports_language(state.project.language_code):
            low_desc = f"{Ansi.ITALICS}Highly recommended when language is not {list(Whitelist.LANGUAGES.keys())}"
            medium_desc = ""
            high_desc = ""
            intolerant_desc = ""
        else:
            low_desc = (
                "Allows for more word errors without triggering a regeneration.\n"
                "      Segments pass unless notably off.")
            medium_desc = "Balanced. Reasonable choice for most TTS models and most languages."
            high_desc = warning_high if warning_high else "Strict; segments with minor word errors will be flagged for regeneration."
            intolerant_desc = (
                "Segments with even one word error are flagged for regeneration.\n"
                "      Will trigger frequent regenerations due to false positives, but yields best net accuracy.\n"
                "      For the time and compute unconstrained only."
            )

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

def make_tolerance_label(state: State) -> str:
    label = make_menu_label(
        label="Word error tolerance",
        value=state.project.strictness.label
    )
    warning = Tts.get_class().get_strictness_warning(state.project.strictness, state.project, None)
    if warning:
        # Add red asterisk
        label += f"{COL_ERROR}*"
    return label

def count_out_of_range_voice_indices(
        phrase_groups: list[PhraseGroup],
        indices: Iterable[int],
        num_voice_values: int,
) -> int:
    """
    Count user-defined voice selections that are stale (past the end of the voice sample list) and will be clamped.
    `voice_index == -1` is the "no explicit assignment" default (the first voice sample), not an out-of-range selection.
    """
    return sum(
        1 for index in indices
        if phrase_groups[index].voice_index >= num_voice_values
    )

def make_retries_label(state: State) -> str:
    return make_menu_label(
        label="Max retries",
        value=state.project.max_retries
    )

def make_limit_silence_gaps_label(state: State) -> str:
    if state.project.limit_silence_gaps:
        value = f"True {state.project.limit_silence_gaps_duration:.2f}s"
    else:
        value = "False"
    return make_menu_label(
        label="Limit silence gaps",
        value=value,
        default=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS
    )

def make_gen_auto_concat_label(state: State) -> str:
    return make_menu_label(
        label="Concatenate when finished",
        value=state.project.gen_auto_concat
    )

def ask_retries(state: State) -> None:
    MenuUtil.print_screen_heading(state, make_retries_label(state))
    printt(RETRIES_DESC)
    ask.ask_number_and_save(
        state.project,
        "max_retries",
        "Enter value:",
        PROJECT_MAX_RETRIES_MIN, PROJECT_MAX_RETRIES_MAX, PROJECT_MAX_RETRIES_DEFAULT, "Max retries set to:", is_int=True
    )

def ask_batch_size(state: State) -> None:

    field_name = Tts.get_type().value.batch_size_attr
    if not field_name:
        return # silently ignore (shouldn't happen)

    hints.show_hint_if_necessary(state.prefs, HINT_BATCH)

    prompt = "Enter max concurrent requests:" if Tts.is_sgl_mode() else "Enter batch size:"

    # Note that if there is a TtsModelType local and server "member pair" for the same underlying TTS model,
    # and the two share the same Project "batch_size" attribute, that value can be out of range
    # compared to the 'correct' max value.
    max_value = PROJECT_CONCURRENT_REQUESTS_MAX if Tts.is_sgl_mode() else PROJECT_BATCH_SIZE_MAX

    ask.ask_number_and_save(
        state.project, field_name, prompt,
        1, max_value, PROJECT_BATCH_SIZE_DEFAULT,
        "Set batch size:", is_int=True
    )

def do_generate(state: State) -> None:

    # Check blockers
    error = readiness.get_generate_blocker_text(state, verbose=True)
    if error:
        print_feedback(error, is_error=True)
        return

    # Get indices to generate, and check if already generated
    indices = ProjectUtil.get_selected_indices_not_generated(state.project)

    if not indices:
        all_lines_generated = (
            state.project.sound_segments.num_generated()
            == len(state.project.phrase_groups)
        )
        message = (
            "All lines already generated"
            if all_lines_generated
            else "No lines queued to be generated"
        )
        print_feedback(message)
        return

    # Show pre-inference hint/warning if necessary
    should_continue = app_hint_util.show_pre_inference_hints(state.prefs, state.project)
    if not should_continue:
        return

    # Print pseudo-menu heading
    MenuUtil.print_screen_heading(state, "Start")

    # Print queue info
    range_string = ProjectUtil.generate_range_string_display(state.project)
    line_word = "line" if len(indices) == 1 else "lines"
    s = f"- Will generate {len(indices)} {line_word} {COL_DIM}({range_string})"
    num = state.project.sound_segments.num_generated_in_current_range()
    if num:
        s += f" {COL_DIM}({num} already complete)"
    printt(s)
    # Print batching setting
    tts_type = Tts.get_type()
    if tts_type.value.can_batch:
        batch_size = ProjectVoiceUtil.get_batch_size(state.project)
        if Tts.is_sgl_mode():
            s = f"- Concurrent requests: {batch_size}"
        else:
            if batch_size > 1:
                s = f"- Batch size: {batch_size}"
            else:
                s = "- Batching: disabled"
        printt(s)
    # Print stt setting
    if not Stt.should_skip(state):
        s = "- Speech-to-text validation: enabled"
        s += f" {COL_DIM}({Stt.short_description()})"
    else:
        s = "- Speech-to-text validation: disabled"
        printt(s)

    # Print voice selection mode info
    voice_values = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
    if len(voice_values) > 1:
        s = "- Voice selection mode: "
        s += f"{state.project.voice_select_mode.label}"
        if state.project.voice_select_mode == VoiceSelectMode.USER_DEFINED:
            num_invalid_voice_indices = count_out_of_range_voice_indices(
                state.project.phrase_groups, indices, len(voice_values)
            )
            if num_invalid_voice_indices:
                selection_word = make_noun(
                    "selection",
                    "selections",
                    num_invalid_voice_indices,
                )
                s += f"\n  {COL_ERROR}Warning: {num_invalid_voice_indices} voice {selection_word} out of range and will be clamped"
        printt(s)
    # Print auto-concat setting
    if state.project.gen_auto_concat:
        printt("- Will concatenate audio file/s when finished")

    # Confirm
    printt()
    b = ask.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
    if not b:
        return

    # Generate in full-screen modal.
    # The model worker remains alive after the modal app exits.
    generation_result = run_generation_modal(
        state=state,
        indices=indices,
        batch_size=ProjectVoiceUtil.get_batch_size(state.project),
        is_regen=False,
    )

    if generation_result.completed and state.project.gen_auto_concat:
        printt()
        ConcatUtil.auto_concat_after_generation(state)

# ---

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

LIMIT_SILENCE_GAPS_MENU_SUBHEADING = \
"""Limits instances of silence within sound segment from extending beyond
a certain duration.

Larger values can be used to prevent long pauses (eg, 1-2 seconds).

Small values can be used to influence pacing and prosody (eg, 0.0-0.3 seconds).
(Consider using \"Text segmentation strategy: Sentence.\" in that case)

This setting also applies to realtime playback, voice chat, stand-alone server.
"""
