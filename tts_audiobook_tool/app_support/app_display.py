from tts_audiobook_tool.app_types import BookSegmentationSettings
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.system_support.terminal import get_terminal_width


def print_text_groups(groups: list[PhraseGroup]) -> None:
    s = f"Text segments ({COL_DIM}{len(groups)}{COL_DEFAULT}):"
    MenuUtil.print_heading(None, s, non_menu=True)
    printt()

    for i, group in enumerate(groups):
        printt(f"{make_hotkey_string(str(i + 1))} {group.presentable_text}")
    printt()


def print_project_text(
    phrase_groups: list[PhraseGroup],
    extant_indices: set[int] | None,
    segmentation_settings: BookSegmentationSettings,
) -> None:
    """
    Prints the list of text segments of a Project.

    extant_indices:
        When exists, prints if sound gen exists for text segment, and prints num-generated info
    """

    heading = "Text segments:" if extant_indices else "Text segments preview:"
    MenuUtil.print_heading(None, heading, dont_clear=True)

    if len(phrase_groups) > 0:
        index_width = len(str(len(phrase_groups)))

        for i, phrase_group in enumerate(phrase_groups):
            index_string = "[" + str(i + 1).rjust(index_width) + "]"
            if extant_indices is not None:
                exists_string = "[" + ("generated" if i in extant_indices else " missing ") + "] "
            else:
                exists_string = ""
            if DEV:
                reason_string = COL_DIM + " [" + phrase_group.as_flattened_phrase().reason.json_value + "]"
            else:
                reason_string = ""
            printt(f"{COL_ACCENT}{index_string} {COL_DIM}{exists_string}{COL_DEFAULT}{phrase_group.presentable_text}{reason_string}")
    else:
        printt("None")
    printt()

    printt(f"{COL_DIM}The text was segmented using the following settings:")
    printt(f"- Text segmenter language code: {COL_ACCENT}{segmentation_settings.language_code or 'none'}")
    if segmentation_settings.max_words_per_segment:
        printt(f"- Text segmenter max_words_per_segment: {COL_ACCENT}{segmentation_settings.max_words_per_segment}")
    printt(f"- Text segmenter strategy: {COL_ACCENT}{segmentation_settings.strategy.label}")
    if extant_indices is not None:
        printt()
        printt(f"Num audio segments generated: {COL_ACCENT}{len(extant_indices)} {COL_DIM}/ {COL_ACCENT}{len(phrase_groups)}")
    printt()


def print_regen_lines(project: Project, indices: set[int]) -> None:
    from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil

    MenuUtil.print_heading(None, "Lines to be regenerated:", non_menu=True, dont_clear=True)

    if not indices:
        printt("None")
        printt()
        return

    for index in sorted(indices):
        SegmentTranscriptUtil.print_info(index, project)

    printt()

def make_terminal_divider(width: int | None = None, char: str = "-") -> str:
    width = width or get_terminal_width()
    return char * max(1, width)
