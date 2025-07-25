from tts_audiobook_tool.app import App
from tts_audiobook_tool.app_types import TtsType
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *

"""
App entrypoint:
- Does prerequisite checks which when failed exits the app
- Prints one-time info messages, depending
"""

print()

err = Tts.init_active_model()
if err:
    ask_error(err)
    exit(1)

# FFMPEG check
if not FfmpegUtil.is_ffmpeg_available():
    printt("The command 'ffmpeg' must exist on the system path.")
    printt("Please install it first.")
    printt("https://ffmpeg.org/download.html")
    exit(1)

# Print some one-time messages
prefs = Prefs.load()

if not is_long_path_enabled():
    AppUtil.show_hint_if_necessary(prefs, HINT_LONG_PATHS, and_prompt=True)

if Tts.get_type() == TtsType.OUTE:
    AppUtil.show_hint_if_necessary(prefs, HINT_OUTE_CONFIG, and_prompt=True)

printt()

# Start proper
app = App()
app.loop()
