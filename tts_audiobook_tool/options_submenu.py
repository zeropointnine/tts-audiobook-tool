from tts_audiobook_tool.real_time import RealTime
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class OptionsSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            print_heading("Options, tools:")
            printt(f"{make_hotkey_string("1")} Generate and play in real time")
            printt(f"{make_hotkey_string("2")} Add app metadata to a pre-existing audiobook file")
            printt(f"{make_hotkey_string("3")} Transcode app-created FLAC to MP4, preserving custom metadata")
            printt()

            hotkey = ask_hotkey()
            match hotkey:
                case "1":
                    RealTime.go(state.project)
                    break
                case "2":
                    SttFlow.ask_and_make(state.prefs)
                    break
                case "3":
                    TranscodeUtil.ask_transcode(state)
                    break
                case _:
                    break
