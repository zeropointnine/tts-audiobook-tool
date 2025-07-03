from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.real_time import RealTime
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *


class RealTimeSubmenu:

    use_custom_text = False
    custom_text_segments: list[TextSegment] = []

    @staticmethod
    def submenu(state: State):

        print_heading("Real-time generation and playback")
        AppUtil.show_hint_if_necessary(state.prefs, "real_time", "About:", HINT_TEXT)
        printt(f"{make_hotkey_string("1")} Start")

        if RealTimeSubmenu.use_custom_text:
            s = f"currently: using custom text, {len(RealTimeSubmenu.custom_text_segments)} lines"
        else:
            s = "currently: using project text"
        printt(f"{make_hotkey_string("2")} Text source {COL_DIM}({s})")
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
                ask_continue("No text segments specified")
                return
            RealTime.start(state.project, text_segments)
        elif hotkey == "2":
            RealTimeSubmenu.text_submenu(state)

    @staticmethod
    def text_submenu(state: State) -> None: # type: ignore

        print_heading("Real time - Set text source")
        printt(f"{make_hotkey_string("1")} Use project text")
        printt(f"{make_hotkey_string("2")} Temporary text - import from text file")
        printt(f"{make_hotkey_string("3")} Temporary text - enter/paste text")
        printt()

        hotkey = ask_hotkey()
        if not hotkey:
            RealTimeSubmenu.submenu(state)
        elif hotkey == "1":
            RealTimeSubmenu.use_custom_text = False
            RealTimeSubmenu.custom_text_segments = state.project.text_segments
            RealTimeSubmenu.submenu(state)
        if hotkey == "2":
            text_segments, _ = AppUtil.get_text_segments_from_ask_text_file()
            if text_segments:
                RealTimeSubmenu.use_custom_text = True
                RealTimeSubmenu.custom_text_segments = text_segments
        if hotkey == "3":
            text_segments, _ = AppUtil.get_text_segments_from_ask_std_in()
            if text_segments:
                RealTimeSubmenu.use_custom_text = True
                RealTimeSubmenu.custom_text_segments = text_segments

        RealTimeSubmenu.submenu(state)


HINT_TEXT = """This uses the same quality-control steps as the normal "Generate" workflow
except for loudness normalization.

For uninterrupted playback, your system must be able to to do the audio inference
faster-than-realtime."""