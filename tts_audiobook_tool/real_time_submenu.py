from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.real_time_util import RealTimeUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class RealTimeSubmenu:

    @staticmethod
    def submenu(state: State):

        while True:

            print_heading("Real-time playback")
            AppUtil.show_hint_if_necessary(state.prefs, HINT_REAL_TIME)
            printt(f"{make_hotkey_string('1')} Start")

            if state.real_time.custom_text_segments:
                value = f"custom text, {len(state.real_time.custom_text_segments)} lines"
            else:
                value = "project text"
            current = make_currently_string(value)
            printt(f"{make_hotkey_string('2')} Text source {COL_DIM} {current}")

            line_range = state.real_time.line_range
            if line_range:
                value = f"{line_range[0]}-{line_range[1]}"
            else:
                value = "all"

            printt(f"{make_hotkey_string('3')} Select line range {make_currently_string(value)}")
            printt()

            hotkey = ask_hotkey()
            if not hotkey:
                break

            match hotkey:
                case "1":
                    if state.real_time.custom_text_segments:
                        text_segments = state.real_time.custom_text_segments
                    else:
                        text_segments = state.project.text_segments
                    if not text_segments:
                        ask_continue("No text segments specified.\n")
                        continue
                    RealTimeUtil.start(state.project, text_segments, line_range)
                case "2":
                    RealTimeSubmenu.text_submenu(state)
                case "3":
                    RealTimeSubmenu.ask_line_range(state)
                case _:
                    break

    @staticmethod
    def ask_line_range(state: State) -> None:

        if state.real_time.custom_text_segments:
            text_segments = state.real_time.custom_text_segments
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

        state.real_time.line_range = result

        # Print feedback
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

        if hotkey == "1":
            if state.real_time.custom_text_segments:
                state.real_time.custom_text_segments = []
                state.real_time.line_range = None
            printt_set("Text source set to: project")

        elif hotkey == "2" or hotkey == "3":
            if hotkey == "2":
                text_segments, _ = AppUtil.get_text_segments_from_ask_text_file()
            else:
                text_segments, _ = AppUtil.get_text_segments_from_ask_std_in()
            if text_segments:
                state.real_time.custom_text_segments = text_segments
                state.real_time.line_range = None
                printt_set(f"Text source set to: custom, {len(text_segments)} lines")
