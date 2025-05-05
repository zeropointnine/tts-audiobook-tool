import json
import os
import sys

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from .text_segmenter import TextSegmenter
from .util import *
from .constants import *

class TextSegmentsUtil:

    @staticmethod
    def ask_text_segments_and_set(state: State, max_words_per_segment: int=MAX_WORDS_PER_SEGMENT) -> None:
        """ Gets text from user and saves to disk and sets State. Or prints error message. """

        printt("Enter/paste text of any length.")
        printt(f"Finish with {COL_ACCENT}[CTRL-Z + ENTER]{COL_DEFAULT} or {COL_ACCENT}[ENTER + CTRL-D]{COL_DEFAULT}, depending on platform")
        text = sys.stdin.read().strip()
        printt()
        if not text:
            return

        texts = TextSegmenter.segment_full_message(text, max_words=max_words_per_segment)
        texts = [item.strip() for item in texts]
        texts = [item for item in texts if item]
        if not texts:
            return

        print_text_segments(texts)
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

        state.text_segments = texts
