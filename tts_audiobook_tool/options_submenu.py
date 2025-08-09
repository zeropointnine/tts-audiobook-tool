from tts_audiobook_tool.mp3_concat import Mp3ConcatTranscodeUtil
from tts_audiobook_tool.real_time_submenu import RealTimeSubmenu
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class OptionsSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            print_heading("Options/Tools:")
            printt(f"{make_hotkey_string("1")} Real-time generation and playback")
            printt(f"{make_hotkey_string("2")} Enhance existing audiobook {COL_DIM}(experimental)")
            printt(f"{make_hotkey_string("3")} Transcode and concatenate a directory of MP3 files to AAC/M4A")
            printt(f"{make_hotkey_string("4")} Transcode an app-created FLAC to AAC/M4A, preserving its custom metadata")
            printt(f"{make_hotkey_string("5")} Reset contextual hints")
            printt()

            hotkey = ask_hotkey()
            match hotkey:
                case "1":
                    RealTimeSubmenu.submenu(state)
                case "2":
                    SttFlow.ask_and_make(state.prefs)
                case "3":
                    Mp3ConcatTranscodeUtil.ask_mp3_dir()
                case "4":
                    TranscodeUtil.ask_transcode_abr_flac_to_aac(state)
                case "5":
                    state.prefs.reset_hints()
                    printt("One-time contextual hints have been reset.\nThey will now appear again when relevant.\n")
                    ask_continue()
                case _:
                    break
