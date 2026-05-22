import os

from tts_audiobook_tool.app_types import BookSegmentationSettings
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool import text_util
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.system_support.terminal import get_terminal_width


def print_book_sections(state: State) -> None:
    
    MenuUtil.print_screen_heading(state, "Sections")

    project = state.project
    sections = project.book.sections
    section_start_indices = ProjectBookUtil.get_section_start_indices(project)

    if len(sections) == 0:
        printt("None")
    else:   
        index_width = len(str(len(sections)))
        for i, section in enumerate(sections):
            
            index_string = "[" + str(i + 1).rjust(index_width) + "]"
            
            printt(f"{COL_ACCENT}{index_string} {COL_DEFAULT}{ellipsize(section.title, length=60)}")

            if not section.phrase_groups:
                continue

            phrase_group_index = section_start_indices[i]
            phrase_group = project.phrase_groups[phrase_group_index]
            printt(f"{COL_DIM}{' ' * (index_width + 3)}Line {phrase_group_index + 1}: {ellipsize(phrase_group.presentable_text, length=60)}")
    printt()

def print_book_text_lines(
    state: State,
    phrase_groups: list[PhraseGroup],
    extant_indices: set[int] | None,
    segmentation_settings: BookSegmentationSettings,
) -> None:
    """
    Prints the list of text segments of a Project.

    state:
        Is used for print_screen_heading only; text data is passed in independently
        (because it may or may not be part of project yet)

    extant_indices:
        When exists, prints if sound gen exists for text segment, and prints num-generated info
    """

    heading = "Text segments" if extant_indices else "Text segments preview"
    MenuUtil.print_screen_heading(state, heading)

    if len(phrase_groups) == 0:
        printt("None")
    
    else:   
        index_width = len(str(len(phrase_groups)))
        for i, phrase_group in enumerate(phrase_groups):
            index_string = "[" + str(i + 1).rjust(index_width) + "]"
            if extant_indices is not None:
                if i in extant_indices:
                    file_name = state.project.sound_segments.get_best_file_for(i)
                    if file_name:
                        file_path = os.path.join(state.project.sound_segments_path, file_name)
                        generated_string = text_util.make_terminal_hyperlink(file_path, "generated", is_file=True)
                    else:
                        generated_string = "generated"
                    exists_string = f"[{generated_string}] "
                else:
                    exists_string = "[ missing ] "
            else:
                exists_string = ""
            if DEV:
                reason_string = COL_DIM + " [" + phrase_group.as_flattened_phrase().reason.json_value + "]"
            else:
                reason_string = ""
            printt(f"{COL_ACCENT}{index_string} {COL_DIM}{exists_string}{COL_DEFAULT}{phrase_group.presentable_text}{reason_string}")

    printt()

    # Print segmentation settings used
    printt(f"{COL_DIM}The text was segmented using the following settings:")
    printt(f"- Text segmenter language code: {COL_ACCENT}{segmentation_settings.language_code or 'none'}")
    if segmentation_settings.max_words_per_segment:
        printt(f"- Text segmenter max_words_per_segment: {COL_ACCENT}{segmentation_settings.max_words_per_segment}")
    printt(f"- Text segmenter strategy: {COL_ACCENT}{segmentation_settings.strategy.label}")
    printt()

    # Print line gen info
    if extant_indices:
        s = f"{COL_DIM}Lines with generated sound segments "
        s += f"{COL_DIM}({COL_ACCENT}{len(extant_indices)} {COL_DIM}/ {COL_ACCENT}{len(phrase_groups)}{COL_DIM})\n"
        s += COL_DEFAULT + RangeStringUtil.make_ranges_string(extant_indices, len(phrase_groups))
        printt(s)
        printt()


def print_regen_lines(state: State, indices: set[int]) -> None:
    from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil

    MenuUtil.print_heading(state, "Lines to be regenerated")

    if not indices:
        printt("None")
        printt()
        return

    for index in sorted(indices):
        SegmentTranscriptUtil.print_info(index, state.project)

    printt()

def make_terminal_divider(width: int | None = None, char: str = "-") -> str:
    width = width or get_terminal_width()
    return char * max(1, width)
