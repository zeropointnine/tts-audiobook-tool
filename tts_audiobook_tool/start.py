from importlib import util
from tts_audiobook_tool.app import App
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *


"""
App entrypoint:
- Does prerequisite checks which when failed exits the app
- Prints one-time info messages, depending
"""

printt()

# TTS model check
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

# Updated dependencies check
new_packages = ["faster_whisper", "pynvml"]
not_found = [package for package in new_packages if not util.find_spec(package)]
if not_found:
    hint = Hint(
        "none", "The app's dependencies have changed",
        f"The following packages were not found: {", ".join(not_found)}\n"
        "You've may have updated the app from the repository without updating its dependencies.\n"
        f"Please update your virtual environment by re-running:\n"
        f"`pip install -r {Tts.get_type().value.requirements_file_name}`."
    )
    AppUtil.print_hint(hint)
    exit(1)

# Print some one-time messages
prefs = Prefs.load()
if not is_long_path_enabled():
    AppUtil.show_hint_if_necessary(prefs, HINT_LONG_PATHS, and_prompt=True)
if Tts.get_type() == TtsType.OUTE:
    AppUtil.show_hint_if_necessary(prefs, HINT_OUTE_CONFIG, and_prompt=True)

# Start proper
printt()
app = App()
app.loop()
