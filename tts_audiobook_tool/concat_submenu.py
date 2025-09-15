import os

from tts_audiobook_tool.app_types import NormalizationType
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class ConcatSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            print_heading(f"Concatenate audio segments:")

            infos = ChapterInfo.make_chapter_infos(state.project)
            if len(infos) > 1:
                print_chapter_segment_info(infos)

            if not state.project.section_dividers:
                chapter_dividers_desc = ""
            else:
                strings = [str(item+1) for item in state.project.section_dividers] # 1-indexed
                chapter_dividers_desc = ", ".join(strings)
                chapter_dividers_desc = f": {COL_ACCENT}{chapter_dividers_desc}{COL_DIM}"

            printt(f"{make_hotkey_string('1')} Create FLAC file")
            printt(f"{make_hotkey_string('2')} Create AAC/M4A file")
            s = "none" if not state.project.section_dividers else f"{len(state.project.section_dividers)} cut point/s"
            printt(f"{make_hotkey_string('3')} Define file cut points {make_currently_string(s)}")

            s = state.prefs.normalization_type.value.label
            printt(f"{make_hotkey_string('4')} Loudness normalization {make_currently_string(s)}")
            printt()

            hotkey = ask_hotkey()
            if hotkey == "1":
                ConcatSubmenu.ask_chapters_and_make(infos, state, to_aac_not_flac=False)
            elif hotkey == "2":
                ConcatSubmenu.ask_chapters_and_make(infos, state, to_aac_not_flac=True)
            elif hotkey == "3":
                ConcatSubmenu.ask_cut_points(state)
            elif hotkey == "4":
                ConcatSubmenu.ask_normalization(state.prefs)
            else:
                break

    @staticmethod
    def ask_normalization(prefs: Prefs) -> None:

        hotkey_to_norm_type: dict[str, NormalizationType] = {}
        hotkey = "1"
        for item in NormalizationType:
            hotkey_to_norm_type[hotkey] = item
            hotkey = chr(ord(hotkey) + 1)

        print_heading("Loudness normalization:")

        if Tts.get_type() == TtsModelInfos.OUTE:
            AppUtil.show_hint_if_necessary(prefs, HINT_OUTE_LOUD_NORM)

        for hotkey, norm_type in hotkey_to_norm_type.items():
            s = f"{make_hotkey_string(hotkey)}"
            printt(f"{s} {norm_type.value.label}")
        printt()

        hotkey = ask_hotkey()
        if hotkey in hotkey_to_norm_type:
            norm_type = hotkey_to_norm_type[hotkey].value
            prefs.set_normalization_type_using(norm_type.json_value)
            printt_set(f"Normalization set to: {norm_type.label}")
            return
        elif hotkey == "":
            return
        else:
            ConcatSubmenu.ask_normalization(prefs)

    @staticmethod
    def ask_cut_points(state: State) -> None:

        print_heading("Combine > File cut points:")

        num_text_segments = len(state.project.text_segments)

        section_dividers = state.project.section_dividers
        if section_dividers:
            print_concat_info(section_dividers, num_text_segments)

        printt("Enter the line numbers where new files will begin.")
        printt("For example, if there are 400 lines of text and you enter \"101, 201\",")
        printt("three audio files will be created spanning lines 1-100, 101-200, and 201-400.")
        printt("Enter \"1\" for no cut points.")
        printt()
        inp = ask()
        if not inp:
            return

        string_items = inp.split(",")
        one_indexed_items = []
        for string_item in string_items:
            try:
                index = int(string_item)
                one_indexed_items.append(index)
            except:
                ask_error(f"Parse error: {string_item}")
                return
        one_indexed_items = list(set(one_indexed_items))
        one_indexed_items.sort()
        for item in one_indexed_items:
            if item < 1 or item > len(state.project.text_segments):
                ask_error(f"Index out of range: {item}")
                return
        zero_indexed_items = [item - 1 for item in one_indexed_items]
        if 0 in zero_indexed_items:
            del zero_indexed_items[0]
        state.project.section_dividers = zero_indexed_items
        state.project.save()

        if not zero_indexed_items:
            s = "none"
        else:
            s = ", ".join( [str(item) for item in zero_indexed_items] )
        printt_set(f"Cut points set to: {s}")

    @staticmethod
    def ask_chapters_and_make(infos: list[ChapterInfo], state: State, to_aac_not_flac: bool) -> None:

        # Chapter indices that have any files
        chapter_indices = []
        for i, info in enumerate(infos):
            if info.num_files_exist > 0:
                chapter_indices.append(i)

        if len(chapter_indices) > 1:
            printt("Enter file numbers for which you want to combine audio segments")
            printt("(For example: \"1, 2, 4\" or  \"2-5\", or \"all\")")
            inp = ask()
            if inp == "all" or inp == "a":
                chapter_indices = chapter_indices.copy()
            else:
                input_indices, warnings = ParseUtil.parse_one_indexed_ranges_string(inp, len(state.project.text_segments))
                if warnings:
                    for warning in warnings:
                        printt(warning)
                    printt()
                    return
                if not input_indices:
                    return
                chapter_indices = [item for item in input_indices if item in chapter_indices]

                if not chapter_indices:
                    ask("No valid chapters numbers entered. Press enter: ")
                    return

        else:
            chapter_indices = [0]

        printt(f"Will concatenate audio segments to create the following chapter file/s:")
        strings = []
        for chapter_index in chapter_indices:
            info = infos[chapter_index]
            s = f"{chapter_index+1} (lines {info.segment_index_start+1}-{info.segment_index_end+1})"
            strings.append(s)
        string = ", ".join(strings)
        printt(string)
        printt()
        b = ask_confirm()
        if not b:
            return

        ConcatSubmenu.make_chapter_files(state, chapter_indices, to_aac_not_flac)

    @staticmethod
    def make_chapter_files(state: State, chapter_indices: list[int], to_aac_not_flac: bool) -> None:

        # Make subdir
        timestamp_subdir = timestamp_string()
        dest_subdir = os.path.join(state.project.dir_path, PROJECT_CONCAT_SUBDIR, timestamp_subdir)
        try:
            os.makedirs(dest_subdir, exist_ok=True)
        except:
            ask_error(f"Couldn't make directory {dest_subdir}")
            return

        for i, chapter_index in enumerate(chapter_indices):

            if len(chapter_indices) > 1:
                s = f" {COL_ACCENT}{i+1}{COL_DEFAULT}/{COL_ACCENT}{len(chapter_indices)}{COL_DEFAULT} - chapter {COL_ACCENT}{chapter_index+1}{COL_DEFAULT}"
            else:
                s = ""
            print_heading(f"Creating concatenated audio file{s}...", dont_clear=True)

            is_norm = (state.prefs.normalization_type != NormalizationType.DISABLED)
            is_concat_aac = not is_norm and to_aac_not_flac

            # Concat
            path, err = ConcatUtil.concatenate_chapter_file(
                state=state,
                chapter_index=chapter_index,
                to_aac_not_flac=is_concat_aac,
                base_dir=dest_subdir
            )
            if err:
                printt()
                ask_error(err)
                return

            # Normalize
            if state.prefs.normalization_type != NormalizationType.DISABLED:

                source_path = path
                norm_path = AppUtil.insert_bracket_tag_file_path(path, "normalized")
                if to_aac_not_flac:
                    norm_path = str( Path(norm_path).with_suffix(".m4a") )

                err = LoudnessNormalizationUtil.normalize_file(
                    path, state.prefs.normalization_type.value, norm_path
                )
                if err:
                    ask_error(err)
                    return
                if not DEV_SAVE_INTERMEDIATE_FILES:
                    delete_silently(source_path)
                path = norm_path

            printt(f"Saved {COL_ACCENT}{path}")
            printt()

        # Post-concat feedback
        printt("Finished. \a")
        printt()

        AppUtil.show_player_hint_if_necessary(state.prefs)

        hotkey = ask_hotkey(f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open output directory: ")
        if hotkey == "o":
            err = open_directory_in_gui(dest_subdir)
            if err:
                ask_error(err)

# ---

def print_concat_info(section_dividers: list[int], num_items: int) -> None:
    section_index_strings = [str(index+1) for index in section_dividers]
    section_indices_string = ", ".join(section_index_strings)
    ranges = make_section_ranges(section_dividers, num_items)
    range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
    ranges_string = ", ".join(range_strings)
    printt(f"Current cut points: {section_indices_string} {COL_DIM}({ranges_string})")
    printt()


def print_chapter_segment_info(infos: list[ChapterInfo]) -> None:
    for i, info in enumerate(infos):
        if info.num_files_missing == 0:
            desc = "all lines generated"
        else:
            desc = f"{info.num_files_exist} of {info.num_segments} lines generated"
        s = f"File {i+1}: lines {info.segment_index_start + 1}-{info.segment_index_end + 1}"
        if desc:
            s += f" {COL_DIM}({desc})"
        printt(s)
    printt()
