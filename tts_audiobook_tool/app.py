from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.main_menu import MainMenu
from tts_audiobook_tool.state import State

class App:
    """
    Main app class
    """

    def __init__(self):

        AppUtil.init_logging()
        SigIntHandler().init()

        self.state = State()

        MainMenu.menu_loop(self.state)
