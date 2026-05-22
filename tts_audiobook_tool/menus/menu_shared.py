from tts_audiobook_tool.app_types import SectionMarkerMode
from tts_audiobook_tool.app_types.output_range_info import OutputRangeInfo
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


def make_output_files_subheading(state: State) -> str:

    if state.project.chapter_mode != SectionMarkerMode.FILES:
        return ""

    infos = OutputRangeInfo.make_output_range_infos(state.project)
    if len(infos) == 1:
        return ""

    if len(infos) > 4:
        subinfos = infos[:3]
        extra = f" {COL_DIM_ITALICS}... {COL_ACCENT}+{len(infos) - len(subinfos)} {COL_DIM}more files"
    else:
        subinfos = infos
        extra = ""

    strings = make_output_range_info_strings(subinfos, list(range(len(subinfos))))
    if extra:
        strings[-1] += extra
    string = "\n".join(strings)
    string += "\n"
    return string


def make_output_range_info_strings(infos: list[OutputRangeInfo], indices: list[int]) -> list[str]:
    lst = []
    for index in indices:
        info = infos[index]
        s = f"{COL_DEFAULT}File {index+1}:{COL_DIM} lines {info.segment_index_start + 1} to {info.segment_index_end + 1} "
        s += f"({info.num_files_exist}/{info.num_segments} generated)"
        lst.append(s)
    return lst