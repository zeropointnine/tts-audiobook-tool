from tts_audiobook_tool import app_support, ask
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.menus.main_menu import MainMenu
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State

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
            ask.ask_error(worker_error)

        try:
            MainMenu.menu_loop(self.state)
        finally:
            ModelWorker.shutdown()
