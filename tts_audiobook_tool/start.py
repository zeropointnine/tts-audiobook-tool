# Suppress console output from oute lib
from loguru import logger
logger.remove()

# On startup, show some feedback because importing deps takes a few seconds
from .constants import *
from .util import *
printt(f"\n{COL_ACCENT}Initializing...{Ansi.RESET}\n")

# FFMPEG check
if not is_ffmpeg_available():
    printt("The command 'ffmpeg' must exist on the system path.")
    printt("Please install it first.")
    printt("https://ffmpeg.org/download.html")
    exit(1)

# Start
from tts_audiobook_tool.app import App
app = App()

from tts_audiobook_tool.prefs_util import PrefsUtil
is_first_run = not os.path.exists(PrefsUtil.get_file_path())
if is_first_run:
    printt("\nThis appears to be your first time running tts-audiobook-tool.")
    printt("As a reminder, you may want to adjust the settings in the file 'model_config.py'")
    printt("to enable hardware acceleration, depending on your system setup.\n")
    ask("Press enter to start:")
elif MENU_CLEAR_SCREEN:
    ask(f"\n{COL_ACCENT}Ready. Press enter to start: ")

app.loop()
