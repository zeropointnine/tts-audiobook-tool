from tts_audiobook_tool.app import App
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.util import *

# FFMPEG check
if not is_ffmpeg_available():
    printt("The command 'ffmpeg' must exist on the system path.")
    printt("Please install it first.")
    printt("https://ffmpeg.org/download.html")
    exit(1)

# First-time prompt
is_first_run = not os.path.exists(Prefs.get_file_path())
if is_first_run:
    printt("\nThis appears to be your first time running tts-audiobook-tool.")
    printt(f"As a reminder, you may want to adjust the settings in the file {COL_ACCENT}tts_config.py")
    printt("to enable hardware acceleration, depending on your system setup.\n")
    ask("Press enter to start: ")
elif MENU_CLEAR_SCREEN:
    ask(f"\n{COL_ACCENT}Ready. Press enter to start: ")
else:
    printt()

# Start
app = App()
app.loop()
