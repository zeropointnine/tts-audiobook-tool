from importlib import util
from tts_audiobook_tool.app import App
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.util import *

"""
App entrypoint:
- Does tts model check
- Does ffmpeg check
- Prints first-time info message re: Oute
"""

# Model type check
has_oute = util.find_spec("outetts") is not None
has_chatterbox = util.find_spec("chatterbox") is not None

if has_oute and has_chatterbox:
    print("Both outetts and chatterbox libraries are installed, which is, at the moment, unadvised.")
    print("Please follow the install instructions in the README.")
    exit(1)
elif not has_oute and not has_chatterbox:
    print("Either outetts or chatterbox-tts must be installed.")
    print("Please follow the install instructions in the README.")
    exit(1)
if has_oute:
    Shared.set_model_type("oute")
else:
    Shared.set_model_type("chatterbox")

# FFMPEG check
if not FfmpegUtil.is_ffmpeg_available():
    printt("The command 'ffmpeg' must exist on the system path.")
    printt("Please install it first.")
    printt("https://ffmpeg.org/download.html")
    exit(1)

# Print some first-time messages
is_first_time = not os.path.exists(Prefs.get_file_path())
if is_first_time:

    if not is_long_path_enabled():
        printt()
        printt("⚠ It appears your system does not support long file paths.")
        printt("  tts-audiobook-tool relies on pretty long filenames, so")
        printt("  please use short directory paths when creating new projects")
        printt()
        ask_continue()

    if Shared.is_oute():
        # TODO need extra pref value for "is first time running oute"
        printt()
        printt("⚠ This appears to be your first time running the application with the Oute TTS model")
        printt()
        printt(f"As a reminder, you may want to adjust/experiment with the settings in the file {COL_ACCENT}config_oute.py")
        printt("to enable hardware acceleration, depending on your system setup.")
        printt()
        ask_continue()

printt()

# Start
app = App()
app.loop()
