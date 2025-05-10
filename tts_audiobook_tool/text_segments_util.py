import json
import os
import sys

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
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
        text = sys.stdin.read().strip()
        printt()
        if not text:
            return

        texts = TextSegmenter.segment_full_message(text, max_words=max_words_per_segment)
        texts = TextSegmentsUtil._post_process(texts)
        if not texts:
            return

        AppUtil.print_text_segments(texts)
        printt("... is how the text will be segmented for inference.\n")
        hotkey = ask_hotkey(f"Enter {make_hotkey_string("Y")} to confirm: ")
        if hotkey != "y":
            return

        # Save to disk
        file_path = os.path.join(state.project_dir, PROJECT_TEXT_FILE_NAME)
        err = AppUtil.save_json(texts, file_path)
        if err:
            printt(err, "error")
            return

        # Save raw text as well
        file_path = os.path.join(state.project_dir, PROJECT_RAW_TEXT_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            L.e(f"Error saving raw text: {e}") # fail silently

        state.text_segments = texts

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