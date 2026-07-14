from __future__ import annotations
import subprocess

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *

"""
Creates M4B chapter metadata and applies it to AAC/MP4 containers.
"""

def make_metadata(
        project: Project, 
        durations: list[float],
        file_title: str,
        artist: str=APP_NAME,
        index_start: int=0,
        index_end: int | None=None,
) -> str:
    """
    Creates M4B chapter metadata string.
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

    chapter_info = make_chapter_info(project, duration_sums, index_start, index_end)

    meta = ";FFMETADATA1\n"
    meta += f"title={file_title}\n"
    meta += f"artist={artist}\n\n"

    for start_time, end_time, chapter_title in chapter_info:
        section  = "[CHAPTER]\n"
        section += "TIMEBASE=1/1000\n"
        section += f"START={int(start_time * 1000)}\n"
        section += f"END={int(end_time * 1000)}\n"
        section += f"title={chapter_title}\n\n"
        meta += section
            
    return meta

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

def make_copy_with_metadata(source_path: str, dest_path: str, metadata: str) -> str:
    """
    Makes a copy of an M4A/M4B/MP4 file with added M4B chapter metadata.
    Using ffmpeg is the most compatible approach for doing this.
    Returns error string if any.
    """
    if not os.path.exists(source_path):
        return f"Source file does not exist: {source_path}"
    
    if not source_path.lower().endswith(tuple(AAC_SUFFIXES)):
        return f"Source file has incorrect filename suffix: {source_path}"
    
    full_command = [
        FFMPEG_COMMAND,
        "-y",                 # Overwrite 
        "-i", source_path,    # Input 0: The original file
        "-i", "-",            # Input 1: Standard input (the 'meta' string)
        "-map", "0",          # Map all streams from source
        "-map_metadata", "0", # Keep the original global metadata
        "-map_chapters", "1", # Take the new chapters from the pipe
        "-c", "copy",         # Copy streams without re-encoding
        "-movflags", "use_metadata_tags", # Preserve custom tags
        dest_path        
    ]

    try:
        # Use 'input' to pass the string to stdin. 
        # 'capture_output' ensures e.stderr is populated on failure.
        subprocess.run(
            full_command,
            input=metadata,
            check=True,
            text=True,
            encoding='utf-8',
            capture_output=True
        )

    except subprocess.CalledProcessError as e:
        err_detail = e.stderr.splitlines()[-1] if e.stderr else "Unknown error"
        return f"Ffmpeg error {e.returncode}: {err_detail}"

    except Exception as e:
        return f"Subprocess error: {str(e)}"
    
    return ""
