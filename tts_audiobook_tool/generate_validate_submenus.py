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

        num_segments = len(state.project.text_segments)
        if not state.project.generate_ints:
            state.project.generate_ints = set(range(num_segments)) # this should happen elsewhere
        ranges_string = ParseUtil.make_one_indexed_ranges_string(state.project.generate_ints, num_segments)

        print_heading("Generate audio")
        printt(f"{make_hotkey_string("1")} Generate")
        printt(f"{make_hotkey_string("2")} Generate, plus validate and attempt fix when necessary")
        printt(f"{make_hotkey_string("3")} Specify text segments to generate {COL_DIM}(current: {COL_ACCENT}{ranges_string}{COL_DIM})")
        printt()
        hotkey = ask_hotkey()

        if hotkey == "3":
            GenerateValidateSubmenus.ask_items(state)
            GenerateValidateSubmenus.generate_submenu(state)
            return

        if not (hotkey == "1" or hotkey == "2"):
            return

        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        already_complete = list(dic.keys())
        original_len = len(state.project.generate_ints)

        indices = [item for item in state.project.generate_ints if item not in already_complete]

        if not indices:
            ask(f"All specified items already generated ({original_len}). Press enter: ")
            return

        info = f"Will generate {len(indices)} audio segment/s"
        num_completed = original_len - len(indices)
        if num_completed > 0:
            info += f" (already complete: {num_completed} items)"
        printt(info)
        printt()

        mode = "generate-and-fix" if hotkey == "2" else "generate"
        GenerateUtil.generate_validate_fix_items(state, indices, mode=mode)


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

        state.project.generate_ints = indices

    @staticmethod
    def validate_submenu(state: State) -> None:

        print_heading("Validate/fix generated audio:")

        index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
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
