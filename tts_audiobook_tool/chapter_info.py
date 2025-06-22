from __future__ import annotations

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.util import *


class ChapterInfo:
    def __init__(self, chapter_index: int, segment_index_start: int, segment_index_end: int, segment_index_to_path: dict[int, str]):
        self.chapter_index = chapter_index
        self.segment_index_start = segment_index_start
        self.segment_index_end = segment_index_end
        self.segment_index_to_path = segment_index_to_path

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

        result = []

        all_segment_index_to_path = ProjectDirUtil.get_items(project)

        segment_index_ranges = make_section_ranges(project.section_dividers, len(project.text_segments))

        for chapter_index, rang in enumerate(segment_index_ranges):

            segment_index_start = rang[0]
            segment_index_end = rang[1]

            segment_index_to_path: dict[int, str] = {}

            for segment_index in range(segment_index_start, segment_index_end + 1):
                segment_file_path = all_segment_index_to_path.get(segment_index, "")
                if segment_file_path:
                    segment_index_to_path[segment_index] = segment_file_path

            chapter_info = ChapterInfo(
                chapter_index = chapter_index,
                segment_index_start=segment_index_start,
                segment_index_end=segment_index_end,
                segment_index_to_path=segment_index_to_path
            )
            result.append(chapter_info)

        return result


