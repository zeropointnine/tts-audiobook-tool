from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool import text_util
from tts_audiobook_tool.util import *
from tts_audiobook_tool.system_support.terminal import get_terminal_width


def print_regen_lines(state: State, indices: set[int]) -> None:
    from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil

    MenuUtil.print_heading(state, "Lines to be regenerated")

    if not indices:
        printt("None")
        printt()
        return

    for index in sorted(indices):
        lines = SegmentTranscriptUtil.make_info_text_lines(index, state.project)
        printt(text_util.combine_ansi_lines(lines))

    printt()

def make_terminal_divider(width: int | None = None, char: str = "-") -> str:
    width = width or get_terminal_width()
    return char * max(1, width)
