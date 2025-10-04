from __future__ import annotations

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class GenerateSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            total_segments_generated = state.project.sound_segments.num_generated()
            s = f"{COL_DIM}({COL_ACCENT}{total_segments_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} total lines complete)"
            print_heading(f"Generate audio {s}")

            printt(f"{make_hotkey_string('1')} Generate audio")

            s1 = f"{COL_DIM}(currently set to generate lines {COL_ACCENT}{state.project.generate_range_string or 'all'}{COL_DIM})"
            selected_indices = state.project.get_indices_to_generate()
            all_generated_indices = state.project.sound_segments.sound_segments.keys()
            selected_indices_not_generated = selected_indices - all_generated_indices
            num_selected_indices_generated = len(selected_indices) - len(selected_indices_not_generated)
            s2 = f"({COL_ACCENT}{num_selected_indices_generated}{COL_DIM} of {COL_ACCENT}{len(selected_indices)}{COL_DIM} complete)"
            s = f"{make_hotkey_string('2')} Specify segments to generate {s1} {s2}"
            printt(s)

            failed_items = state.project.sound_segments.get_sound_segments_with_tag("fail")
            s = f"{str(len(failed_items))} {'item' if len(failed_items) == 1 else 'items'}"
            s = make_currently_string(s)
            printt(f"{make_hotkey_string('3')} Regenerate audio segments tagged as having potential errors {s}")
            printt()

            hotkey = ask_hotkey()
            if not hotkey:
                return

            match hotkey:
                case "1":
                    if state.project.can_voice and state.project.get_voice_label() == "none":
                        should_continue = AppUtil.show_hint_if_necessary(state.prefs, HINT_NO_VOICE, and_confirm=True)
                        if not should_continue:
                            continue
                    GenerateSubmenu.do_generate_items(state)
                case "2":
                    GenerateSubmenu.ask_items(state)
                case "3":
                    GenerateSubmenu.do_regenerate_items(state)
                case _:
                    break

    @staticmethod
    def do_generate_items(state: State) -> None:

        selected_indices_all = state.project.get_indices_to_generate()
        selected_indices_generated = set( state.project.sound_segments.sound_segments.keys() )
        selected_indices_not_generated = selected_indices_all - selected_indices_generated

        if not selected_indices_not_generated:
            ask(f"All items in specified range already generated. Press enter: ")
            return

        s = f"Generating {len(selected_indices_not_generated)} audio segment/s..."
        print_heading(s, dont_clear=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        did_interrupt = GenerateUtil.generate_items_to_files(
            project=state.project,
            indices_to_generate=selected_indices_not_generated,
            items_to_regenerate={}
        )

        if did_interrupt:
            ask_continue()
        else:
            s = f"Press enter or {make_hotkey_string('C')} to concatenate files now: \a"
            hotkey = ask_hotkey(s)
            if hotkey == "c":
                ConcatSubmenu.submenu(state)
        return

    @staticmethod
    def do_regenerate_items(state: State) -> None:

        failed_items = state.project.sound_segments.get_sound_segments_with_tag("fail")
        if not failed_items:
            ask_continue("No failed items to regenerate.")
            return

        print_heading(f"Regenerating {len(failed_items)} audio segment/s...", dont_clear=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        will_hint = not state.prefs.get_hint("regenerate")
        AppUtil.show_hint_if_necessary(state.prefs, HINT_REGEN)
        if will_hint:
            b = ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
            if not b:
                return

        _ = GenerateUtil.generate_items_to_files(
            project=state.project,
            indices_to_generate=set(),
            items_to_regenerate=failed_items
        )

        ask_continue()

    @staticmethod
    def ask_items(state: State) -> None:

        num_items = len(state.project.text_segments)

        printt("Enter line numbers to generate (eg, \"1-100, 103\", or \"all\")")
        inp = ask()
        if inp == "all" or inp == "a":
            indices = set( [item for item in range(0, num_items)] )
        else:
            indices, warnings = ParseUtil.parse_one_indexed_ranges_string(inp, num_items)
            if not indices:
                return
            if warnings:
                printt_set("\n".join(warnings))
                return

        s = ParseUtil.make_one_indexed_ranges_string(indices, len(state.project.text_segments))
        state.project.generate_range_string = s
        state.project.save()

        printt_set(f"Line numbers set: {s}")

