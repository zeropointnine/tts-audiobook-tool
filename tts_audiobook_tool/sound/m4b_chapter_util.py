from __future__ import annotations

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.audio_writers import Chapter
from tts_audiobook_tool.util import *

"""
Computes M4B chapters for an output range. The AAC writer writes them into
the container in the same pass as the audio (see AacWriter.set_chapters).
"""

def make_chapters(
        project: Project,
        durations: list[float],
        index_start: int=0,
        index_end: int | None=None,
) -> list[Chapter]:
    """
    Chapters for the output range [index_start, index_end] (inclusive).

    `durations` holds the final rendered duration of each phrase group in the
    range, starting at index_start (so durations[0] belongs to group index_start).
    Returns an empty list unless the range has more than one chapter.
    """

    sum = 0
    duration_sums = []
    for i in range(len(durations)):
        duration_sums.append(sum)
        sum += durations[i]
        if i == len(durations) - 1:
            duration_sums.append(sum) # Add end item

    if index_end is None:
        index_end = index_start + len(durations) - 1

    return [
        Chapter(start=start_time, end=end_time, title=chapter_title)
        for start_time, end_time, chapter_title in make_chapter_info(project, duration_sums, index_start, index_end)
    ]

def has_multiple_chapters(project: Project, index_start: int, index_end: int) -> bool:
    """
    Returns whether this output range contains more than one Book section, which is
    the condition for writing M4B chapter metadata.
    """
    return len(make_output_chapter_sections(project, index_start, index_end)) > 1

def make_output_sections(project: Project, index_start: int, index_end: int) -> list[tuple[int, int, str]]:
    """
    Returns Book sections overlapping the inclusive phrase-group output range as
    tuples of absolute start index, absolute end index (exclusive), and title.
    """
    sections = project.book.sections
    if len(sections) <= 1:
        return []

    starts = project.book.section_start_indices()
    result: list[tuple[int, int, str]] = []
    for section_index, section in enumerate(sections):
        section_start = starts[section_index]
        section_end = starts[section_index + 1] if section_index + 1 < len(starts) else section_start + len(section.phrase_groups)
        overlaps = section_start <= index_end and section_end > index_start
        if overlaps:
            result.append((section_start, section_end, section.title))
    return result

def make_output_chapter_sections(project: Project, index_start: int, index_end: int) -> list[tuple[int, int, str]]:
    """
    Returns the effective chapter sections for M4B chapter metadata.

    For structurally sectioned books (eg, EPUB), preserves Book-section titles.
    For single-section books, falls back to user-configured section markers so
    plain-text projects can emit chapter metadata in "Adds metadata" mode.
    """
    output_sections = make_output_sections(project, index_start, index_end)
    if len(output_sections) > 1:
        return output_sections

    if not project.markers:
        return []

    result: list[tuple[int, int, str]] = []
    for chapter_start, chapter_end in make_file_line_ranges(project.markers, len(project.phrase_groups)):
        overlaps = chapter_start <= index_end and chapter_end >= index_start
        if not overlaps:
            continue

        title = project.phrase_groups[chapter_start].presentable_text if project.phrase_groups else ""
        result.append((chapter_start, chapter_end + 1, title))
    return result

def make_chapter_info(
        project: Project,
        duration_sums: list[float],
        index_start: int,
        index_end: int,
) -> list[tuple[float, float, str]]:
    output_sections = make_output_chapter_sections(project, index_start, index_end)
    if len(output_sections) <= 1:
        return []

    chapter_info: list[tuple[float, float, str]] = []
    for i, (section_start, _, chapter_title) in enumerate(output_sections):
        relative_start_index = max(section_start, index_start) - index_start
        if not (0 <= relative_start_index < len(duration_sums)):
            continue
        start_time = duration_sums[relative_start_index]

        if i == len(output_sections) - 1:
            end_time = duration_sums[-1]
        else:
            next_section_start = output_sections[i + 1][0]
            relative_end_index = max(next_section_start, index_start) - index_start
            if not (0 <= relative_end_index < len(duration_sums)):
                continue
            end_time = duration_sums[relative_end_index]

        chapter_title = ellipsize(chapter_title, 60)
        info = (start_time, end_time, chapter_title)
        chapter_info.append(info)
    return chapter_info
