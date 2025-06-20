from __future__ import annotations

from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.validate_util import ValidateItem, ValidateUtil

class GenerateValidateSubmenus:

    @staticmethod
    def generate_submenu(state: State) -> None:

        while True:

            num_segments_project = len(state.project.text_segments)

            gen_range_string = state.project.generate_range_string
            is_all = not gen_range_string or gen_range_string == "all" or gen_range_string == "a"
            if is_all:
                gen_range_ints = set(range(num_segments_project))
            else:
                result = ParseUtil.parse_one_indexed_ranges_string(gen_range_string, num_segments_project)
                gen_range_ints = result[0]

            num_generated = ProjectDirUtil.num_generated_in_set(state, gen_range_ints)

            print_heading("Generate audio")
            printt(f"{make_hotkey_string("1")} Generate, plus validate and attempt fix when necessary")
            printt(f"{make_hotkey_string("2")} Generate only")
            s = f"{make_hotkey_string("3")} Specify audio segments to generate {COL_DIM}"
            s += f"(currently: {COL_ACCENT}{gen_range_string or "all"}{COL_DIM}) "
            s += f"({num_generated} of {len(gen_range_ints)} complete)"
            printt(s)
            s = f"{make_hotkey_string("4")} Normalize audio after generate "
            s += f"(currently: {state.prefs.should_normalize})"
            printt(s)
            printt(f"{make_hotkey_string("5")} Play audio after each segment is generated (currently: {state.prefs.play_on_generate})")

            printt()
            hotkey = ask_hotkey()

            match hotkey:

                case value if value in ["1", "2"]:

                    mode = "generate-and-fix" if hotkey == "1" else "generate"

                    dic = ProjectDirUtil.get_indices_and_paths(state)
                    already_complete = list(dic.keys())
                    indices = [item for item in gen_range_ints if item not in already_complete]
                    if not indices:
                        ask(f"All specified items already generated. Press enter: ")
                        return

                    printt(f"Generating {len(indices)} audio segment/s...")
                    printt(f"{COL_DIM}Press control-c to interrupt")
                    printt()
                    GenerateUtil.generate_validate_fix_items(state, indices, mode=mode)
                    return

                case "3":
                    GenerateValidateSubmenus.ask_items(state)

                case "4":
                    state.prefs.should_normalize = not state.prefs.should_normalize
                    printt(f"Set to: {state.prefs.should_normalize}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()

                case "5":
                    state.prefs.play_on_generate = not state.prefs.play_on_generate
                    printt(f"Set to: {state.prefs.play_on_generate}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()

                case _:
                    return

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

    @staticmethod
    def validate_submenu(state: State) -> None:

        print_heading("Validate/fix generated audio:")

        index_to_path = ProjectDirUtil.get_indices_and_paths(state)
        all_indices = list( index_to_path.keys() )

        if not all_indices:
            ask("No generated items. Press enter: ")
            return

        # Filter out items with "[pass]" in their filename (ie, already-verified audio files)
        indices = [ index for index in all_indices if not "[pass]" in index_to_path[index] ]

        if not indices:
            ask("All generated audio files already validated. Press enter:" )
            return

        printt(f"Total lines: {len(all_indices)}")
        num_already_verified = len(all_indices) - len(indices)
        if num_already_verified:
            printt(f"Lines already marked as verified: {num_already_verified}")
        printt(f"Lines left to be analysed: {len(indices)}")
        num_missing = len(state.project.text_segments) - len(all_indices)
        if num_missing > 0:
            printt(f"Lines in project that still need to be generated: {num_missing}")
        printt()
        printt(f"{make_hotkey_string("1")} Validate only")
        printt(f"{make_hotkey_string("2")} Validate, and attempt fix when necessary")
        printt()

        hotkey = ask_hotkey()
        if hotkey == "1":
            # Make 'ValidateItem' list
            items = []
            for index in indices:
                text_segment = state.project.text_segments[index]
                file_path = index_to_path[index]
                item = ValidateItem(index, file_path, text_segment.text)
                items.append(item)
            ValidateUtil.validate_items(items)
        elif hotkey == "2":
            GenerateUtil.generate_validate_fix_items(state, indices, "validate-and-fix")
        else:
            return

        ask("Press enter: ")
