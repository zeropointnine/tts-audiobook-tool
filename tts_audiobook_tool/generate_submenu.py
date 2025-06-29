from __future__ import annotations

from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class GenerateSubmenu:

    @staticmethod
    def generate_submenu(state: State) -> None:

        while True:

            indices_to_generate = state.project.get_indices_to_generate()

            num_generated = ProjectDirUtil.count_num_generated_in(state.project, indices_to_generate)

            total_segments_generated = ProjectDirUtil.num_generated(state.project)

            # total_generated

            #s = f"({num_generated} of {len(indices_to_generate)} lines complete)"

            s = f"{COL_DIM}({COL_ACCENT}{total_segments_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} lines complete)"
            print_heading(f"Generate audio {s}")

            s = f"{COL_DIM}(currently set to generate lines: {COL_ACCENT}{state.project.generate_range_string or "all"}{COL_DIM}) "
            printt(f"{make_hotkey_string("1")} Generate {s}")
            s = f"{make_hotkey_string("2")} Specify audio segments to generate"
            printt(s)
            s = f"{make_hotkey_string("3")} Play audio after each segment is generated "
            s += f"{COL_DIM}(currently: {COL_ACCENT}{state.prefs.play_on_generate}{COL_DIM})"
            printt(s)

            printt()
            hotkey = ask_hotkey()

            match hotkey:
                case "1":
                    GenerateSubmenu.do_generate_items(state)
                case "2":
                    GenerateSubmenu.ask_items(state)
                case "3":
                    state.prefs.play_on_generate = not state.prefs.play_on_generate
                    printt(f"Set to: {state.prefs.play_on_generate}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()
                case _:
                    return

    @staticmethod
    def do_generate_items(state: State) -> None:

        all_indices = state.project.get_indices_to_generate()
        already_generated = set(ProjectDirUtil.get_items(state.project).keys())
        not_yet_generated = all_indices - already_generated
        if not not_yet_generated:
            ask(f"All specified items already generated. Press enter: ")
            return

        printt(f"Generating {len(not_yet_generated)} audio segment/s...")
        printt(f"{COL_DIM}Press control-c to interrupt")
        printt()

        did_interrupt = GenerateUtil.generate_items(
            project=state.project,
            indices_to_generate=not_yet_generated,
            items_to_regenerate={},
            play_on_save=state.prefs.play_on_generate
        )

        if did_interrupt:
            ask("Press enter: \a")
        else:
            hotkey = ask_hotkey(f"Press enter or {make_hotkey_string("C")} to concatenate files now: ")
            if hotkey == "c":
                ConcatSubmenu.submenu(state)
        return

    @staticmethod
    def do_regenerate_items(state: State) -> None:

        failed_items = ProjectDirUtil.get_items_with_tag(state.project, "fail")
        if not failed_items:
            ask_continue("No failed items to regenerate")
            return

        printt(f"Regenerating {len(failed_items)} audio segment/s...")
        printt(f"{COL_DIM}Press control-c to interrupt")
        printt()

        _ = GenerateUtil.generate_items(
            project=state.project,
            indices_to_generate=set(),
            items_to_regenerate=failed_items,
            play_on_save=state.prefs.play_on_generate
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
            if warnings:
                printt("\n".join(warnings))
                printt()
                return
            if not indices:
                return

        s = ParseUtil.make_one_indexed_ranges_string(indices, len(state.project.text_segments))
        state.project.generate_range_string = s
        state.project.save()
