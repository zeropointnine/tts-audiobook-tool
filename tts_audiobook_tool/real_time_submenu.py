from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.real_time import RealTime
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *


class RealTimeSubmenu:

    use_custom_text = False
    custom_text_segments: list[TextSegment] = []
    start_index: int = 0

    @staticmethod
    def submenu(state: State):

        # TODO add start-at option here, which should invalidate when text is replace

        print_heading("Real-time playback")
        AppUtil.show_hint_if_necessary(state.prefs, HINT_REAL_TIME)
        printt(f"{make_hotkey_string('1')} Start")

        if RealTimeSubmenu.use_custom_text:
            s = f"custom text, {len(RealTimeSubmenu.custom_text_segments)} lines"
        else:
            s = "project text"
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('2')} Text source {COL_DIM} {s}")
        s = make_currently_string(str(RealTimeSubmenu.start_index + 1))
        printt(f"{make_hotkey_string('3')} Start at line number {s}")
        printt()

        hotkey = ask_hotkey()
        if not hotkey:
            return

        if hotkey == "1":
            if RealTimeSubmenu.use_custom_text:
                text_segments = RealTimeSubmenu.custom_text_segments
            else:
                text_segments = state.project.text_segments
            if not text_segments:
                ask_continue("No text segments specified.\n")
                RealTimeSubmenu.submenu(state)
                return
            RealTime.start(state.project, text_segments, RealTimeSubmenu.start_index)
            RealTimeSubmenu.submenu(state)
            return

        elif hotkey == "2":
            RealTimeSubmenu.text_submenu(state)
            RealTimeSubmenu.submenu(state)

        elif hotkey == "3":
            RealTimeSubmenu.ask_start_line_number(state)
            RealTimeSubmenu.submenu(state)

    @staticmethod
    def ask_start_line_number(state: State) -> None:
        inp = ask("Line number to start at: ")
        if not inp:
            return
        try:
            line_number = int(inp)
        except:
            ask_continue("Bad value.")
            return

        if RealTimeSubmenu.use_custom_text:
            text_segments = RealTimeSubmenu.custom_text_segments
        else:
            text_segments = state.project.text_segments
        length = len(text_segments)

        if line_number < 1 or line_number > length:
            ask_continue("Out of range")
            return

        RealTimeSubmenu.start_index = line_number - 1


    @staticmethod
    def text_submenu(state: State) -> None: # type: ignore

        print_heading("Real time - Set text source")
        printt(f"{make_hotkey_string('1')} Use project text")
        printt(f"{make_hotkey_string('2')} Custom text - from text file")
        printt(f"{make_hotkey_string('3')} Custom text - manual input")
        printt()

        hotkey = ask_hotkey()
        if not hotkey:
            return

        if hotkey == "1" and RealTimeSubmenu.use_custom_text:
            RealTimeSubmenu.use_custom_text = False
            RealTimeSubmenu.custom_text_segments = state.project.text_segments
            RealTimeSubmenu.start_index = 0
        elif hotkey == "2":
            text_segments, _ = AppUtil.get_text_segments_from_ask_text_file()
            if text_segments:
                RealTimeSubmenu.use_custom_text = True
                RealTimeSubmenu.custom_text_segments = text_segments
                RealTimeSubmenu.start_index = 0
        elif hotkey == "3":
            text_segments, _ = AppUtil.get_text_segments_from_ask_std_in()
            if text_segments:
                RealTimeSubmenu.use_custom_text = True
                RealTimeSubmenu.custom_text_segments = text_segments
                RealTimeSubmenu.start_index = 0
