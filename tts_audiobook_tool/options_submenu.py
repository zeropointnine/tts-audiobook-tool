from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *


class OptionsSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            print_heading("Options, tools:")
            printt(f"{make_hotkey_string("1")} Create tts-audiobook-tool metadata for a pre-existing audiobook")
            printt(f"{make_hotkey_string("2")} Transcode FLAC to MP4, preserving tts-audiobook-tool metadata")
            printt()
            hotkey = ask_hotkey()

            match hotkey:
                case "1":
                    SttFlow.ask_and_make(state.prefs)
                    break
                case "2":
                    TranscodeUtil.ask_transcode(state)
                    break
                case _:
                    break
