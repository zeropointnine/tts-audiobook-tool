import os
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.main_menu import MainMenu
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.words_dict import Dictionary

class App:
    """
    Main app class.
    Runs a loop that prints menu, responds to menu selection.
    """

    def __init__(self):

        AppUtil.init_logging()
        SigIntHandler().init()
        Dictionary.init()

        self.state = State()

    def loop(self):
        while True:
            # Dir check # TODO refactor this out
            did_reset = False
            if self.state.prefs.project_dir and not os.path.exists(self.state.prefs.project_dir):
                self.state.reset()
                did_reset = True

            MainMenu.menu(self.state, did_reset=did_reset)
