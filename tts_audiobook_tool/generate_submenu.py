from __future__ import annotations

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class GenerateSubmenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_start(_, __) -> None:
            if AskUtil.is_readchar:
                b = AskUtil.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
                if not b:
                    return
            AppUtil.show_inference_hints(state.prefs, state.project)
            GenerateSubmenu.do_generate_items(state)

        def make_range_label(_) -> str:
            if not state.project.generate_range_string:
                range_label = f"{COL_DIM}(currently set to generate {COL_ACCENT}all{COL_DIM} lines)"
            else:
                range_label = f"{COL_DIM}(currently set to generate lines {COL_ACCENT}{state.project.generate_range_string}{COL_DIM})"

            if not state.project.generate_range_string:
                complete_label = ""
            else:
                selected_indices = state.project.get_indices_to_generate()
                all_generated_indices = state.project.sound_segments.sound_segments.keys()
                selected_indices_not_generated = selected_indices - all_generated_indices
                num_selected_indices_generated = len(selected_indices) - len(selected_indices_not_generated)
                complete_label = f"({COL_ACCENT}{num_selected_indices_generated}{COL_DIM} of {COL_ACCENT}{len(selected_indices)}{COL_DIM} complete)"

            return f"Specify range {range_label} {complete_label}"

        def make_regen_label(_) -> str:
            failed_items = state.project.sound_segments.get_failed_in_generate_range()
            failed_items_label = f"{str(len(failed_items))} {make_noun('item', 'items', len(failed_items))}"
            qualifier = " in specified range" if state.project.generate_range_string else ""
            regenerate_label = f"{COL_DIM}(currently: {COL_ACCENT}{failed_items_label}{COL_DIM}{qualifier})"
            return f"Regenerate segments tagged with potential errors {regenerate_label}"

        # Menu
        def heading_maker(_) -> str:
            total_segments_generated = state.project.sound_segments.num_generated()
            num_complete_label = f"{COL_DIM}({COL_ACCENT}{total_segments_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} total lines complete)"
            return f"Generate audio {num_complete_label}"

        items = [
            MenuItem("Generate audio segments", on_start),
            MenuItem(make_range_label, lambda _, __: GenerateSubmenu.ask_item_range(state)),
            MenuItem(make_regen_label, lambda _, __: GenerateSubmenu.do_regenerate_items(state))
        ]
        MenuUtil.menu(state, heading_maker, items)

    @staticmethod
    def do_generate_items(state: State) -> None:

        selected_indices_all = state.project.get_indices_to_generate()
        selected_indices_generated = set( state.project.sound_segments.sound_segments.keys() )
        selected_indices_not_generated = selected_indices_all - selected_indices_generated

        if not selected_indices_not_generated:
            AskUtil.ask_enter_to_continue(f"All items in specified range already generated.")
            return

        s = f"Generating {len(selected_indices_not_generated)} audio segment/s..."
        if state.prefs.stt_variant == SttVariant.DISABLED:
            s += f" {COL_DIM}(speech-to-text validation disabled){COL_ACCENT}"

        print_heading(s, dont_clear=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        did_interrupt = GenerateUtil.generate_items_to_files(
            project=state.project,
            indices_to_generate=selected_indices_not_generated,
            items_to_regenerate={},
            stt_variant=state.prefs.stt_variant,
            stt_config=state.prefs.stt_config
        )

        if did_interrupt:
            AskUtil.ask_enter_to_continue()
        else:
            s = f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('C')} to concatenate files now: \a"
            hotkey = AskUtil.ask_hotkey(s)
            printt() # TODO revisit
            if hotkey == "c":
                ConcatSubmenu.menu(state)
        return

    @staticmethod
    def do_regenerate_items(state: State) -> None:

        failed_items = state.project.sound_segments.get_failed_in_generate_range()
        if not failed_items:
            qualifier = " in specified range" if state.project.generate_range_string else ""
            print_feedback(f"No failed items to regenerate{qualifier}.")
            return

        print_heading(f"Regenerating {len(failed_items)} audio segment/s...", dont_clear=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        will_hint = not state.prefs.get_hint("regenerate")
        AppUtil.show_hint_if_necessary(state.prefs, HINT_REGEN)
        if will_hint:
            b = AskUtil.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
            if not b:
                return

        _ = GenerateUtil.generate_items_to_files(
            project=state.project,
            indices_to_generate=set(), # rem, ignored
            items_to_regenerate=failed_items,
            stt_variant=state.prefs.stt_variant,
            stt_config=state.prefs.stt_config
        )

        AskUtil.ask_enter_to_continue()

    @staticmethod
    def ask_item_range(state: State) -> None:

        num_items = len(state.project.text_segments)

        printt("Enter line numbers to generate (eg, \"1-100, 103\", or \"all\")")
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

        s = ParseUtil.make_ranges_string(indices, len(state.project.text_segments))
        state.project.generate_range_string = "" if s == "all" else s
        state.project.save()

        print_feedback(f"Range set to: {s}")
