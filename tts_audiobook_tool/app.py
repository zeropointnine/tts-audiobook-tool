import os
from pathlib import Path
import signal
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.main_menu_util import MainMenuUtil
from tts_audiobook_tool.options_util import OptionsUtil
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.generate_validate_submenus import GenerateValidateSubmenus
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.text_segments_util import TextSegmentsUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_util import VoiceUtil

class App:
    """
    Main app class.
    Runs a loop that prints menu, responds to menu selection.
    """

    def __init__(self):

        AppUtil.init_logging()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.state = State()

    def signal_handler(self, _, __):

        def print_message(s: str):
            printt()
            printt(COL_ERROR + "*" * len(s))
            printt(s)
            printt(COL_ERROR + "*" * len(s))
            printt()

        match Shared.mode:
            case "generating":
                Shared.stop_flag = True
                print_message("Control-C pressed, will stop after current gen...")
            case "validating":
                Shared.stop_flag = True
                print_message("Control-C pressed, will stop")
            case "menu":
                Shared.stop_flag = True

    def loop(self):
        while True:

            # Dir check # TODO refactor this out
            did_reset = False
            if self.state.prefs.project_dir and not os.path.exists(self.state.prefs.project_dir):
                self.state.reset()
                did_reset = True

            MainMenuUtil.menu(self.state, did_reset=did_reset)
