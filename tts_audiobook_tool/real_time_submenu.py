from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *


class RealTimeSubmenu:

    custom_text_segments: list[TextSegment] = []

    @staticmethod
    def submenu(state: State):

        print_heading("Generate and play in real time")
        printt(f"{make_hotkey_string("1")} Start")
        s = ""
        printt(f"{make_hotkey_string("2")} Text {s}")

        hotkey = ask_hotkey()
        if not hotkey:
            return
        if hotkey == "1":
            ...
        elif hotkey == "2":
            ...


    def text_submenu(state: State) -> None: # type: ignore

        print_heading("Real time - Set text")
        printt(f"{make_hotkey_string("1")} Use project text")
        printt(f"{make_hotkey_string("2")} Temporary text - import from text file")
        printt(f"{make_hotkey_string("3")} Temporary text - enter/paste text")

        hotkey = ask_hotkey()
        if not hotkey:
            RealTimeSubmenu.submenu(state)
            return
        if hotkey == "1":
            RealTimeSubmenu.custom_text_segments = []
            RealTimeSubmenu.submenu(state)
            return
        if hotkey == "2":
            ...



    # Real time - Set text
    # --------------------
    # [1] Use project text (10000 lines)
    # [2] Import from text file
    # [3] Manually enter/paste text




HINT = """⚠ Note:

For uninterrupted playback, your system must be able to to do the audio inference
faster-than-realtime.

This uses the same quality-control steps as the normal "Generate" workflow
except for loudness normalization. Like when using the normal "Generate" workflow,
it will attempt to re-generate voice lines that do not validate so long as there is
enough buffered data to do so without creating an interruption."""