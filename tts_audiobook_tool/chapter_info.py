from __future__ import annotations
from dataclasses import dataclass

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *


@dataclass
class ChapterInfo:
    """
    Stores mappings of project phrase group index to sound segment path within a certain start/end range
    """
    
    chapter_index: int
    segment_index_start: int
    segment_index_end: int
    segment_index_to_path: dict[int, str]

    def __str__(self) -> str:
        return f"<ChapterInfo - index {self.chapter_index} start {self.segment_index_start} end {self.segment_index_end} num exist {self.num_files_exist} num missing {self.num_files_missing}>"

    @property
    def num_segments(self) -> int:
        return self.segment_index_end - self.segment_index_start + 1

    @property
    def num_files_exist(self) -> int:
        return len(self.segment_index_to_path)

    @property
    def num_files_missing(self) -> int:
        return self.num_segments - self.num_files_exist

    @staticmethod
    def make_chapter_infos(project: Project) -> list[ChapterInfo]:
        """
        Makes ChapterInfo objects based on the project's section_dividers
        """
        ranges = make_chapter_ranges(project.section_dividers, len(project.phrase_groups))
        
        result = []
        for chapter_index, rng in enumerate(ranges):
            chapter_info = ChapterInfo(
                chapter_index = chapter_index,
                segment_index_start=rng[0],
                segment_index_end=rng[1],
                segment_index_to_path=make_index_to_path(project, rng[0], rng[1])
            )
            result.append(chapter_info)
        return result

    @staticmethod
    def make_single_info(project: Project) -> ChapterInfo:
        """
        Makes single ChapterInfo, disregarding section_dividers
        """
        index_end = len(project.phrase_groups) - 1
        chapter_info = ChapterInfo(
            chapter_index = 0,
            segment_index_start=0,
            segment_index_end=index_end,
            segment_index_to_path=make_index_to_path(project, 0, index_end)
        )
        return chapter_info

# ---

def make_index_to_path(project: Project, index_start: int, index_end: int) -> dict[int, str]:

    index_to_path: dict[int, str] = {}
    for index in range(index_start, index_end + 1):
        path = project.sound_segments.get_best_file_for(index)
        if path:
            index_to_path[index] = path
    return index_to_path

