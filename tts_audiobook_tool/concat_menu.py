import os

from tts_audiobook_tool.app_types import ExportType, NormalizationType
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
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class ConcatMenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_start(_: State, __: MenuItem) -> None:
            infos = ChapterInfo.make_chapter_infos(state.project)
            is_aac = (state.project.export_type == ExportType.AAC)
            ConcatMenu.ask_chapters_and_make(infos, state, aac_not_flac=is_aac)

        def make_cuts_label(_: State) -> str:
            qty = len(state.project.section_dividers)
            if state.project.section_dividers:
                s = make_currently_string(f"{qty} cut {make_noun('point', 'points', qty)}")
            else:
                s = f"{COL_DIM}(optional)"
            return f"Define chapter cut points {s}"

        items = [
            MenuItem("Start", on_start),
            MenuItem(make_cuts_label, lambda _, __: ConcatMenu.ask_cut_points(state)),
            MenuItem(
                lambda _: make_menu_label("File type", state.project.export_type.label), 
                lambda _, __: ConcatMenu.file_type_menu(state)
            ),
            MenuItem(
                lambda _: make_menu_label("Loudness normalization", state.project.normalization_type.value.label), 
                lambda _, __: ConcatMenu.normalization_menu(state)
            ),
            MenuItem(
                lambda _: make_menu_label("Subdivide into phrases", state.project.subdivide_phrases), 
                lambda _, __: ConcatMenu.subdivide_menu(state)
            ),
            MenuItem(
                lambda _: make_menu_label("Section break sound effect", state.project.use_section_sound_effect),
                lambda _, __: ConcatMenu.section_break_menu(state)
            )
        ]
        MenuUtil.menu(state, "Concatenate audio segments:", items, subheading=make_chapter_info_subheading)

    @staticmethod
    def file_type_menu(state: State) -> None:

        def on_select(value: ExportType) -> None:
            state.project.export_type = value
            state.project.save()
            print_feedback(f"File type set to: {value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text="File type",
            labels=[item.label for item in list(ExportType)],
            values=[item for item in list(ExportType)],
            current_value=state.project.export_type,
            default_value=list(ExportType)[0],
            on_select=on_select
        )

    @staticmethod
    def normalization_menu(state: State) -> None:

        def on_select(value: NormalizationType) -> None:
            state.project.normalization_type = value
            state.project.save()
            print_feedback(f"Normalization set to: {value.value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Loudness normalization",
            labels=[item.value.label for item in list(NormalizationType)],
            values=[item for item in list(NormalizationType)],
            current_value=state.project.normalization_type,
            default_value=list(NormalizationType)[0],
            on_select=on_select,
            subheading=LOUDNORM_SUBHEADING,
            hint=HINT_OUTE_LOUD_NORM if Tts.get_type() == TtsModelInfos.OUTE else None
        )

    @staticmethod
    def subdivide_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.subdivide_phrases = value
            state.project.save()
            print_feedback(f"Subdivide into phrases set to: {state.project.subdivide_phrases}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Subdivide into phrases",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.subdivide_phrases,
            default_value=False,
            on_select=on_select,
            subheading=LOUDNORM_SUBHEADING,
            hint=HINT_OUTE_LOUD_NORM if Tts.get_type() == TtsModelInfos.OUTE else None
        )

    @staticmethod
    def section_break_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.use_section_sound_effect = value
            state.project.save()
            print_feedback(f"Set to:", value)

        MenuUtil.options_menu(
            state=state,
            heading_text="Section break sound effect",
            subheading=SECTION_BREAK_SUBHEADING,
            labels=["False", "True"],
            values=[False, True],
            current_value=state.project.use_section_sound_effect,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def ask_cut_points(state: State) -> None:

        print_heading("Chapter file cut points:")

        num_text_groups = len(state.project.phrase_groups)

        section_dividers = state.project.section_dividers
        if section_dividers:
            print_cut_points(section_dividers, num_text_groups)

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
            if item < 1 or item > len(state.project.phrase_groups):
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
                input_indices, warnings = ParseUtil.parse_ranges_string(inp, len(state.project.phrase_groups))
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

        ConcatMenu.make_chapter_files(state, chapter_indices, aac_not_flac)

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
            print_heading(f"Creating concatenated audio file{s}...", dont_clear=True, non_menu=True)

            is_norm = (state.project.normalization_type != NormalizationType.DISABLED)
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
            if state.project.normalization_type != NormalizationType.DISABLED:

                source_path = path
                norm_path = AppUtil.insert_bracket_tag_file_path(path, "normalized")
                if to_aac_not_flac:
                    norm_path = str( Path(norm_path).with_suffix(".m4a") )

                err = LoudnessNormalizationUtil.normalize_file(
                    path, state.project.normalization_type.value, norm_path
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

        Hint.show_player_hint_if_necessary(state.prefs)

        hotkey = AskUtil.ask_hotkey(f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open output directory in system file browser: ")
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

LOUDNORM_SUBHEADING = \
"""Performs an extra pass after concatenating audio segments to minimize volume disparities
between TTS generations. The \"Stronger\" profile is a more aggressive setting, suitable 
for mobile devices.
"""

SUBDIVIDE_SUBHEADING = \
"""Affects how text is highlighted in the player/reader app.
When False, text segments map directly to the TTS prompts of the generated audio. 
When True, text is further sub-segmented by phrase. Requires \"speech-to-text validation\"
to be enabled during TTS sound generation.
"""

SECTION_BREAK_SUBHEADING = \
"""In the concatenation step, inserts a 'page turn' sound effect when 
two or more consecutive blank lines are encountered in the text. 
This can be a useful audible cue, so long as the text is formatted for it.
"""
