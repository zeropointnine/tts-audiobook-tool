from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        s = str(len(state.project.text_segments))
        s += " line" if len(state.project.text_segments) == 1 else " lines"
        s = make_currently_string(s)
        print_heading(f"Text {s}")
        printt(f"{make_hotkey_string('1')} Replace text")
        printt(f"{make_hotkey_string('2')} View text lines")
        printt()

        hotkey = ask()

        if hotkey == "1":
            num_files = state.project.sound_segments.num_generated()
            if num_files == 0:
                TextSubmenu.set_text_submenu(state, "Replace text:")
            else:
                s = f"Replacing text will invalidate all ({num_files}) previously generated sound segment files for this project.\nAre you sure? "
                if ask_hotkey(s):
                    TextSubmenu.set_text_submenu(state, "Replace text:")
                else:
                    TextSubmenu.submenu(state)
        elif hotkey == "2":
            print_project_text(state)
            ask("Press enter: ")
            TextSubmenu.submenu(state)

    @staticmethod
    def set_text_submenu(state: State, heading: str) -> None:

        print_heading(heading)
        AppUtil.show_hint_if_necessary(state.prefs, HINT_LINE_BREAKS)
        printt(f"{make_hotkey_string('1')} Import from text file")
        printt(f"{make_hotkey_string('2')} Manually enter/paste text")
        printt()

        inp = ask_hotkey()
        if inp == "1":
            text_segments, raw_text = AppUtil.get_text_segments_from_ask_text_file()
            TextSubmenu._finish_set_text(state, text_segments, raw_text)
        elif inp == "2":
            text_segments, raw_text = AppUtil.get_text_segments_from_ask_std_in()
            TextSubmenu._finish_set_text(state, text_segments, raw_text)
        TextSubmenu.submenu(state)

    @staticmethod
    def _finish_set_text(state: State, text_segments: list[TextSegment], raw_text: str) -> None:

        # Print text segments
        strings = [item.text for item in text_segments]
        AppUtil.print_text_segment_text(strings)
        printt("... is how the text will be segmented for inference.\n")

        # Ask for confirmation
        b = ask_confirm()
        if not b:
            return

        # Delete now-outdated gens
        old_sound_segments = state.project.sound_segments.sound_segments
        for path in old_sound_segments.values():
            delete_silently(path)

        # Commit
        state.project.set_text_segments_and_save(text_segments, raw_text=raw_text)

        if not state.real_time.custom_text_segments:
            state.real_time.line_range = None

        printt_set("Project text has been set")

# ---

def print_project_text(state: State) -> None:

    indices = state.project.sound_segments.sound_segments.keys()
    text_segments = state.project.text_segments

    print_heading(f"Text segments ({COL_DEFAULT}{len(text_segments)}{COL_ACCENT}):")

    max_width = len(str(len(text_segments)))

    for i, text_segment in enumerate(text_segments):
        s1 = make_hotkey_string( str(i+1).rjust(max_width) )
        s2 = make_hotkey_string("y" if i in indices else "n")
        printt(f"{s1} {s2}  {text_segment.text.strip()}")
    printt()

    s = ParseUtil.make_ranges_string(set(indices), len(text_segments))
    printt(f"Generated segments: {s}")
    printt()
