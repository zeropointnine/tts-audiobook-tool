import sys

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segmenter import TextSegmenter
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextSegmentsUtil:

    @staticmethod
    def set_text_submenu(state: State) -> None:
        """
        Gets text from user, either by asking for text file path or direct input,
        and saves to disk and sets State. Or prints error message.
        """
        printt(f"{make_hotkey_string("1")} Import text file")
        printt(f"{make_hotkey_string("2")} Manually enter/paste text")
        printt()
        inp = ask_hotkey()
        if inp == "1":
            TextSegmentsUtil.ask_text_import_and_set(state)
        elif inp == "2":
            TextSegmentsUtil.ask_text_input_and_set(state)

    @staticmethod
    def ask_text_import_and_set(state: State) -> None:
        path = ask("Enter text file path: ")
        if not path:
            return
        if not os.path.exists(path):
            printt("No such file", "error")
            return

        try:
            with open(path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
        except Exception as e:
            printt(f"Error: {e}")
            return

        TextSegmentsUtil._finish_set_text(state, raw_text)


    @staticmethod
    def ask_text_input_and_set(state: State) -> None:
        printt("Enter/paste text of any length.")
        printt(f"Finish with {COL_ACCENT}[CTRL-Z + ENTER]{COL_DEFAULT} or {COL_ACCENT}[ENTER + CTRL-D]{COL_DEFAULT} on its own line, depending on platform\n")
        raw_text = sys.stdin.read().strip()
        printt()
        if raw_text:
            TextSegmentsUtil._finish_set_text(state, raw_text)

    @staticmethod
    def _finish_set_text(state: State, raw_text: str) -> None:

        text_segments = TextSegmenter.segment_text(raw_text, max_words=DEFAULT_MAX_WORDS_PER_SEGMENT)

        # Filter out items w/o 'vocalizable content'
        text_segments = [item for item in text_segments if TextSegmentsUtil._has_alpha_numeric(item.text)]

        if not text_segments:
            return

        strings = [item.text for item in text_segments]
        AppUtil.print_text_segment_text(strings)
        printt("... is how the text will be segmented for inference.\n")

        hotkey = ask_hotkey(f"Enter {make_hotkey_string("Y")} to confirm: ")
        if hotkey != "y":
            return

        # Commit
        state.project.set_text_segments_and_save(text_segments, raw_text=raw_text)

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