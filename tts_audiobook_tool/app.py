from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.constants import COL_ERROR
from tts_audiobook_tool.menus.main_menu import MainMenu
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import printt

class App:
    """
    Main app class
    """

    def __init__(self):

        app_support.init_logging()

        Interrupts().init()

        self.state = State()

        worker_error = ModelWorker.start()
        if worker_error:
            printt(f"{COL_ERROR}{worker_error}")

        try:
            MainMenu.menu_loop(self.state)
        finally:
            ModelWorker.shutdown()
