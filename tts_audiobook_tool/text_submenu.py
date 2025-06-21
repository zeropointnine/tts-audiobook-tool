import sys

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segmenter import TextSegmenter
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        print_heading("Text:")
        printt(f"{make_hotkey_string("1")} View text")
        printt(f"{make_hotkey_string("2")} Replace text\n")
        hotkey = ask()
        if hotkey == "1":
            print_project_text(state)
            ask("Press enter: ")
        elif hotkey == "2":
            num_files = ProjectDirUtil.num_generated(state)
            if num_files == 0:
                TextSubmenu.set_text_submenu(state, "Replace text:")
            else:
                s = f"Replacing text will invalidate {num_files} previously generated audio file fragments for this project.\nAre you sure? "
                if ask_hotkey(s):
                    TextSubmenu.set_text_submenu(state, "Replace text:")

    @staticmethod
    def set_text_submenu(state: State, heading: str) -> None:

        print_heading(heading)
        printt(f"{make_hotkey_string("1")} Import text file")
        printt(f"{make_hotkey_string("2")} Manually enter/paste text")
        printt()

        if not state.prefs.has_set_any_text:
            printt("⚠ Note: Line breaks are treated as paragraph delimiters.")
            printt("        If your text uses manual line breaks for formatting (eg, Project Gutenberg files), ")
            printt("        you will want to reformat it first.")
            printt()

        inp = ask_hotkey()
        if inp == "1":
            TextSubmenu.ask_text_import_and_set(state)
        elif inp == "2":
            TextSubmenu.ask_text_input_and_set(state)

    @staticmethod
    def ask_text_import_and_set(state: State) -> None:
        path = ask_path("Enter text file path: ")
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

        TextSubmenu._finish_set_text(state, raw_text)


    @staticmethod
    def ask_text_input_and_set(state: State) -> None:
        printt("Enter/paste text of any length.")
        printt(f"Finish with {COL_ACCENT}[CTRL-Z + ENTER]{COL_DEFAULT} or {COL_ACCENT}[ENTER + CTRL-D]{COL_DEFAULT} on its own line, depending on platform\n")
        raw_text = sys.stdin.read().strip()
        printt()
        if raw_text:
            TextSubmenu._finish_set_text(state, raw_text)

    @staticmethod
    def _finish_set_text(state: State, raw_text: str) -> None:

        state.prefs.has_set_any_text = True

        text_segments = TextSegmenter.segment_text(raw_text, max_words=MAX_WORDS_PER_SEGMENT)

        # Filter out items w/o 'vocalizable content'
        text_segments = [item for item in text_segments if TextSubmenu._has_alpha_numeric_char(item.text)]

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
        lines = [line for line in lines if TextSubmenu._has_alpha_numeric_char(line)]
        return lines

    @staticmethod
    def _has_alpha_numeric_char(s: str) -> bool:
        return any(c.isalnum() for c in s)

# ---

def print_project_text(state: State) -> None:

    index_to_path = ProjectDirUtil.get_indices_and_paths(state)
    indices = index_to_path.keys()
    texts = [item.text for item in state.project.text_segments]

    print_heading(f"Text segments ({COL_DEFAULT}{len(texts)}{COL_ACCENT}):")

    max_width = len(str(len(texts)))

    for i, text in enumerate(texts):
        s1 = make_hotkey_string( str(i+1).rjust(max_width) )
        s2 = make_hotkey_string("y" if i in indices else "n")
        printt(f"{s1} {s2}  {text.strip()}")
    printt()

    indices = set( list( ProjectDirUtil.get_indices_and_paths(state).keys() ) )
    s = ParseUtil.make_one_indexed_ranges_string(indices, len(texts))
    printt(f"Generated segments: {s}")
    printt()
