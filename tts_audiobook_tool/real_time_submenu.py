from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.real_time import RealTime
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *


class RealTimeSubmenu:

    use_custom_text = False
    custom_text_segments: list[TextSegment] = []
    line_range: tuple[int, int] = (0, 0)

    @staticmethod
    def submenu(state: State):

        while True:

            print_heading("Real-time playback")
            AppUtil.show_hint_if_necessary(state.prefs, HINT_REAL_TIME)
            printt(f"{make_hotkey_string('1')} Start")

            if RealTimeSubmenu.use_custom_text:
                value = f"custom text, {len(RealTimeSubmenu.custom_text_segments)} lines"
            else:
                value = "project text"
            current = make_currently_string(value)
            printt(f"{make_hotkey_string('2')} Text source {COL_DIM} {current}")

            line_range = RealTimeSubmenu.line_range
            if line_range[0] == 0 and line_range[1] == 0:
                value = "all"
            else:
                value = f"{RealTimeSubmenu.line_range[0]}-{RealTimeSubmenu.line_range[1]}"
            printt(f"{make_hotkey_string('3')} Select line range {make_currently_string(value)}")
            printt()

            hotkey = ask_hotkey()
            if not hotkey:
                break

            match hotkey:
                case "1":
                    if RealTimeSubmenu.use_custom_text:
                        text_segments = RealTimeSubmenu.custom_text_segments
                    else:
                        text_segments = state.project.text_segments
                    if not text_segments:
                        ask_continue("No text segments specified.\n")
                        continue
                    RealTime.start(state.project, text_segments, line_range)
                case "2":
                    RealTimeSubmenu.text_submenu(state)
                case "3":
                    RealTimeSubmenu.ask_line_range(state)
                case _:
                    break

    @staticmethod
    def ask_line_range(state: State) -> None:

        if RealTimeSubmenu.use_custom_text:
            text_segments = RealTimeSubmenu.custom_text_segments
        else:
            text_segments = state.project.text_segments
        length = len(text_segments)

        s = "Enter line range (eg, \"5-15\"; \"50\" for 50 to end; or \"all\")"
        printt(s)
        inp = ask()
        if not inp:
            return
        result = ParseUtil.parse_range_string_normal(inp, length)
        if isinstance(result, str):
            ask_error(result)
            return

        RealTimeSubmenu.line_range = result

        is_all = (result[0] == 0 and result[1] == 0) or (result[0] == 1 and result[1] == len(text_segments))
        if is_all:
            value = f"1-{len(text_segments)} (all)"
        else:
            value = f"{result[0]}-{result[1]}"
            if result[1] == len(text_segments):
                value += " (end)"
        printt_set(f"Line range set to: {value}")


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
            if not RealTimeSubmenu.use_custom_text:
                return
            RealTimeSubmenu.use_custom_text = False
            RealTimeSubmenu.custom_text_segments = state.project.text_segments
            RealTimeSubmenu.line_range = (0, 0)
        elif hotkey == "2":
            text_segments, _ = AppUtil.get_text_segments_from_ask_text_file()
            if text_segments:
                RealTimeSubmenu.use_custom_text = True
                RealTimeSubmenu.custom_text_segments = text_segments
                RealTimeSubmenu.line_range = (0, 0)
        elif hotkey == "3":
            text_segments, _ = AppUtil.get_text_segments_from_ask_std_in()
            if text_segments:
                RealTimeSubmenu.use_custom_text = True
                RealTimeSubmenu.custom_text_segments = text_segments
                RealTimeSubmenu.line_range = (0, 0)
