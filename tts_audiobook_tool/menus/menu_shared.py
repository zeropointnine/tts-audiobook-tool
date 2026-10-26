from tts_audiobook_tool.app_types import SectionMarkerMode
from tts_audiobook_tool.app_types.output_range_info import OutputRangeInfo
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


def get_num_output_files(state: State) -> int:
    """Number of audiobook files the concat step produces with the current markers."""
    if state.project.chapter_mode != SectionMarkerMode.FILES:
        return 1
    return len(make_file_line_ranges(state.project.markers, len(state.project.phrase_groups)))


def make_concat_menu_heading(state: State) -> str:
    noun = "files" if get_num_output_files(state) > 1 else "file"
    return f"Create audiobook {noun}"


def make_output_files_subheading(state: State, truncate: bool = True) -> str:
    """
    Lists the output files' line ranges and generation progress.
    When `truncate` is set, long lists show only the first few files.
    """

    if state.project.chapter_mode != SectionMarkerMode.FILES:
        return ""

    infos = OutputRangeInfo.make_output_range_infos(state.project)
    if len(infos) == 1:
        return ""

    if truncate and len(infos) > 4:
        subinfos = infos[:3]
        extra = f" {COL_DIM_ITALICS}... {COL_ACCENT}+{len(infos) - len(subinfos)} {COL_DIM}more files"
    else:
        subinfos = infos
        extra = ""

    strings = make_output_range_info_strings(subinfos, list(range(len(subinfos))))
    if extra:
        strings[-1] += extra
    return "\n".join(strings)


def make_full_output_files_subheading(state: State) -> str:
    """Lists every output file, without truncation."""
    return make_output_files_subheading(state, truncate=False)


def make_output_range_info_strings(infos: list[OutputRangeInfo], indices: list[int]) -> list[str]:
    lst = []
    for index in indices:
        info = infos[index]
        s = f"{COL_ACCENT}{BULLET_CHAR} {COL_DIM}File {index+1}: {COL_DEFAULT}lines {info.segment_index_start + 1} to {info.segment_index_end + 1} "
        s += f"({info.num_files_exist}/{info.num_segments} generated)"
        lst.append(s)
    return lst