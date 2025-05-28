# from tts_audiobook_tool.app_util import AppUtil
# from tts_audiobook_tool.loudness_util import LoudnessUtil
# AppUtil.init_logging()
# LoudnessUtil.normalize_directory(r"C:\workspace\main example\segments")
# exit(0)


from tts_audiobook_tool.app import App
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.util import *

# FFMPEG check
if not FfmpegUtil.is_ffmpeg_available():
    printt("The command 'ffmpeg' must exist on the system path.")
    printt("Please install it first.")
    printt("https://ffmpeg.org/download.html")
    exit(1)

# First-time prompt
is_first_run = not os.path.exists(Prefs.get_file_path())
if is_first_run:
    printt()
    printt("This appears to be your first time running tts-audiobook-tool.")
    printt(f"As a reminder, you may want to adjust/experiment with the settings in the file {COL_ACCENT}tts_config.py")
    printt("to enable hardware acceleration, depending on your system setup.")
    printt()
    ask("Press enter to start: ")

printt()

# Start
app = App()
app.loop()
