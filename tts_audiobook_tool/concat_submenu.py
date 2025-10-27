import os

from tts_audiobook_tool.app_types import NormalizationType
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfo, TtsModelInfos
from tts_audiobook_tool.util import *

class ConcatSubmenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_start(_, data):
            infos = ChapterInfo.make_chapter_infos(state.project)
            ConcatSubmenu.ask_chapters_and_make(infos, state, aac_not_flac=data)

        def make_cuts_label(_) -> str:
            qty = len(state.project.section_dividers)
            if state.project.section_dividers:
                s = make_currently_string(f"{qty} cut {make_noun('point', 'points', qty)}")
            else:
                s = f"{COL_DIM}(optional)"
            return f"Define chapter cut points {s}"

        def make_norm_label(_) -> str:
            value = state.prefs.normalization_type.value.label
            return f"Loudness normalization {make_currently_string(value)}"

        items = [
            MenuItem("Create as FLAC file", on_start, data=False),
            MenuItem("Create as AAC/M4A file",on_start, data=True),
            MenuItem(make_cuts_label, lambda _, __: ConcatSubmenu.ask_cut_points(state)),
            MenuItem(make_norm_label, lambda _, __: ConcatSubmenu.normalization_menu(state))
        ]
        MenuUtil.menu(state, "Concatenate audio segments:", items, subheading=make_chapter_info_subheading)

    @staticmethod
    def normalization_menu(state: State) -> None:

        def on_select(_, item: MenuItem) -> None:
            info: NormalizationType = item.data
            state.prefs.set_normalization_type_using(info.value.json_value)
            print_feedback(f"Normalization set to: {info.value.label}")

        menu_items = []
        for typ in list(NormalizationType):
            menu_items.append( MenuItem(typ.value.label, on_select, data=typ) )

        MenuUtil.menu(
            state=state,
            heading="Loudness normalization:",
            items=menu_items,
            hint=HINT_OUTE_LOUD_NORM if Tts.get_type() == TtsModelInfos.OUTE else None,
            one_shot=True
        )

    @staticmethod
    def ask_cut_points(state: State) -> None:

        print_heading("Chapter file cut points:")

        num_text_segments = len(state.project.text_segments)

        section_dividers = state.project.section_dividers
        if section_dividers:
            print_cut_points(section_dividers, num_text_segments)

        printt("Enter the line numbers where new chapter files will begin.")
        printt(f"{COL_DIM}For example, if there are 400 lines of text and you enter \"101, 201\",")
        printt(f"{COL_DIM}three audio files will be created spanning lines 1-100, 101-200, and 201-400.")
        printt(f"{COL_DIM}Enter \"{COL_DEFAULT}1{COL_DIM}\" for no cut points.")
        printt()
        inp = AskUtil.ask()
        if not inp:
            return

        string_items = inp.split(",")
        one_indexed_items = []
        for string_item in string_items:
            try:
                index = int(string_item)
                one_indexed_items.append(index)
            except:
                AskUtil.ask_error(f"Parse error: {string_item}")
                return
        one_indexed_items = list(set(one_indexed_items))
        one_indexed_items.sort()
        for item in one_indexed_items:
            if item < 1 or item > len(state.project.text_segments):
                AskUtil.ask_error(f"Index out of range: {item}")
                return
        zero_indexed_items = [item - 1 for item in one_indexed_items]
        if 0 in zero_indexed_items:
            del zero_indexed_items[0]
        state.project.section_dividers = zero_indexed_items
        state.project.save()

        if not zero_indexed_items:
            s = "none"
        else:
            s = ", ".join( [str(item + 1) for item in zero_indexed_items] )
        print_feedback(f"Cut points set:", s)

    @staticmethod
    def ask_chapters_and_make(infos: list[ChapterInfo], state: State, aac_not_flac: bool) -> None:

        # Chapter indices that have any generated files
        chapter_indices = []
        for i, info in enumerate(infos):
            if info.num_files_exist > 0:
                chapter_indices.append(i)

        if len(chapter_indices) > 1:
            printt("Enter chapter file numbers to create:")
            printt(f"{COL_DIM}(For example: \"1, 2, 4\" or  \"2-5\", or \"all\")")
            inp = AskUtil.ask()
            if inp == "all" or inp == "a":
                chapter_indices = chapter_indices.copy()
            else:
                input_indices, warnings = ParseUtil.parse_ranges_string(inp, len(state.project.text_segments))
                if warnings:
                    for warning in warnings:
                        printt(warning)
                    printt()
                    return
                if not input_indices:
                    return
                chapter_indices = [item for item in input_indices if item in chapter_indices]

                if not chapter_indices:
                    AskUtil.ask_enter_to_continue("No valid chapters numbers entered.")
                    return

        s = f"Will create the following {'AAC' if aac_not_flac else 'FLAC'} {make_noun('file', 'files', len(chapter_indices))}:"
        printt(s)
        strings = make_chapter_info_strings(infos, chapter_indices)
        s ="\n".join( ("    " + item) for item in strings)
        printt(s)
        printt()

        b = AskUtil.ask_confirm()
        if not b:
            return

        ConcatSubmenu.make_chapter_files(state, chapter_indices, aac_not_flac)

    @staticmethod
    def make_chapter_files(state: State, chapter_indices: list[int], to_aac_not_flac: bool) -> None:

        # Make subdir
        timestamp_subdir = timestamp_string()
        dest_subdir = os.path.join(state.project.dir_path, PROJECT_CONCAT_SUBDIR, timestamp_subdir)
        try:
            os.makedirs(dest_subdir, exist_ok=True)
        except:
            AskUtil.ask_error(f"Couldn't make directory {dest_subdir}")
            return

        for i, chapter_index in enumerate(chapter_indices):

            if len(chapter_indices) > 1:
                s = f" {COL_ACCENT}{i+1}{COL_DEFAULT}/{COL_ACCENT}{len(chapter_indices)}{COL_DEFAULT} - chapter file {COL_ACCENT}{chapter_index+1}{COL_DEFAULT}"
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
                AskUtil.ask_error(err)
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
                    AskUtil.ask_error(err)
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

        hotkey = AskUtil.ask_hotkey(f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open output directory: ")
        printt()
        if hotkey == "o":
            err = open_directory_in_gui(dest_subdir)
            if err:
                AskUtil.ask_error(err)

# ---

def print_cut_points(section_dividers: list[int], num_items: int) -> None:
    section_index_strings = [str(index+1) for index in section_dividers]
    section_indices_string = ", ".join(section_index_strings)
    ranges = make_section_ranges(section_dividers, num_items)
    range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
    ranges_string = ", ".join(range_strings)
    printt(f"Current cut points: {section_indices_string} {COL_DIM}({ranges_string})")
    printt()

def make_chapter_info_subheading(state: State) -> str:
    infos = ChapterInfo.make_chapter_infos(state.project)
    if len(infos) == 1:
        return ""
    strings = make_chapter_info_strings( infos, list(range(len(infos))) )
    string = "\n".join(strings)
    string += "\n" # extra line
    return string

def make_chapter_info_strings(infos: list[ChapterInfo], indices: list[int]) -> list[str]:
    lst = []
    for index in indices:
        s = make_chapter_info_string(infos[index], index)
        lst.append(s)
    return lst

def make_chapter_info_string(info: ChapterInfo, index: int) -> str:
    s = f"{COL_DEFAULT}Chapter file {index+1}:{COL_DIM} line {info.segment_index_start + 1} to {info.segment_index_end + 1} "
    s += f"({info.num_files_exist}/{info.num_segments} generated){COL_DEFAULT}"
    return s

