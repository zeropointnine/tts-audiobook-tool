from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *

# Suppress console output from oute lib
from loguru import logger
logger.remove()

# Suppress console output from pyloud
import warnings
warnings.filterwarnings("ignore", module="pyloud")

# FFMPEG check
if not is_ffmpeg_available():
    printt("The command 'ffmpeg' must exist on the system path.")
    printt("Please install it first.")
    printt("https://ffmpeg.org/download.html")
    exit(1)

# Show some feedback because importing deps takes a few seconds
printt(f"\n{COL_ACCENT}Initializing...{Ansi.RESET}\n")


from tts_audiobook_tool.app import App
app = App()

from tts_audiobook_tool.prefs import Prefs
is_first_run = not os.path.exists(Prefs.get_file_path())
if is_first_run:
    printt("\nThis appears to be your first time running tts-audiobook-tool.")
    printt(f"As a reminder, you may want to adjust the settings in the file {COL_ACCENT}model_config.py")
    printt("to enable hardware acceleration, depending on your system setup.\n")
    ask("Press enter to start:")
elif MENU_CLEAR_SCREEN:
    ask(f"\n{COL_ACCENT}Ready. Press enter to start: ")
else:
    printt()
    printt()

app.loop()
