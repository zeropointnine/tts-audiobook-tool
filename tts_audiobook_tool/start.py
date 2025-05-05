# Suppress console output from oute lib
import time
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

if MENU_CLEAR_SCREEN:
    printt("\nReady...")
    time.sleep(1.5)

app.loop()
