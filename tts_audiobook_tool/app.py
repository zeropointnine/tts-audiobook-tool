from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.menus.main_menu import MainMenu
from tts_audiobook_tool.state import State

class App:
    """
    Main app class
    """

    def __init__(self):

        app_support.init_logging()

        Interrupts().init()

        self.state = State()

        MainMenu.menu_loop(self.state)
