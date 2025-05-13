import json
import os
import sys

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segmenter import TextSegmenter
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextSegmentsUtil:

    @staticmethod
    def ask_text_segments_and_set(state: State, max_words_per_segment: int=MAX_WORDS_PER_SEGMENT) -> None:
        """ Gets text from user and saves to disk and sets State. Or prints error message. """

        printt("Enter/paste text of any length.")
        printt(f"Finish with {COL_ACCENT}[CTRL-Z + ENTER]{COL_DEFAULT} or {COL_ACCENT}[ENTER + CTRL-D]{COL_DEFAULT}, depending on platform\n")
        raw_text = sys.stdin.read().strip()
        printt()
        if not raw_text:
            return

        text_segments = TextSegmenter.segment_full_message(raw_text, max_words=max_words_per_segment)
        text_segments = TextSegmentsUtil._post_process(text_segments)
        if not text_segments:
            return

        AppUtil.print_text_segments(text_segments)
        printt("... is how the text will be segmented for inference.\n")
        hotkey = ask_hotkey(f"Enter {make_hotkey_string("Y")} to confirm: ")
        if hotkey != "y":
            return

        # Commit
        state.project.set_text_segments(text_segments, raw_text=raw_text)

    @staticmethod
    def _post_process(lines: list[str]) -> list[str]:
        # Strip
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        # Filter out any lines that do not have at least one alpha/numeric char
        lines = [line for line in lines if TextSegmentsUtil._has_alpha_numeric(line)]
        return lines

    @staticmethod
    def _has_alpha_numeric(s: str) -> bool:
        return any(c.isalnum() for c in s)