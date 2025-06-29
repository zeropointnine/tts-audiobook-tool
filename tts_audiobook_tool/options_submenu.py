from tts_audiobook_tool.generate_submenu import GenerateSubmenu
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class OptionsSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            failed_items = state.project.sound_segments.get_sound_segments_with_tag("fail")

            print_heading("Options, tools:")
            printt(f"{make_hotkey_string("1")} Regenerate voice lines tagged as having potential errors {COL_DIM}(currently: {COL_ACCENT}{len(failed_items)}{COL_DIM} file/s)")
            printt(f"{make_hotkey_string("2")} Create tts-audiobook-tool metadata for a pre-existing audiobook")
            printt(f"{make_hotkey_string("3")} Transcode FLAC to MP4, preserving tts-audiobook-tool metadata")
            printt()

            hotkey = ask_hotkey()
            match hotkey:
                case "1":
                    GenerateSubmenu.do_regenerate_items(state)
                    break
                case "2":
                    SttFlow.ask_and_make(state.prefs)
                    break
                case "3":
                    TranscodeUtil.ask_transcode(state)
                    break
                case _:
                    break
