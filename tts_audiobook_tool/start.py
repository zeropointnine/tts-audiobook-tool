# --------------------------------------------------------------------------------------------------
# Must be imported first or else HF_HUB_CACHE can result in returning a relative path (!)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
from huggingface_hub import constants # type: ignore
# --------------------------------------------------------------------------------------------------

import sys
from importlib import util
from tts_audiobook_tool.app import App
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

def main() -> None:

    """
    App entrypoint:
    - Does prerequisite checks and exits on fail
    - Prints one-time info messages as needed
    - Starts the app proper
    """

    AppUtil.init_logging()
    printt()

    # Hard requirement - FFMPEG 
    if not FfmpegUtil.is_ffmpeg_available():
        printt(f"{COL_ERROR}The command 'ffmpeg' must exist on the system path.")
        printt(f"{COL_ERROR}Please install it first:")
        printt("https://ffmpeg.org/download.html")
        exit(1)

    # Hard requirement - TTS model 
    err = Tts.init_model_type()
    if err:
        printt(f"{COL_ERROR}{err}")
        exit(1)

    # Hard requirement - chatterbox + Python v3.11
    if Tts.get_type() == TtsModelInfos.CHATTERBOX:
        if sys.version_info.major>= 3 and sys.version_info.minor > 11:
            AppUtil.show_hint(HINT_CHATTERBOX_PYTHON_DOWNGRADE)
            exit(1)

    # Updated dependencies check (required)
    not_found = [package for package in NEW_PACKAGES if not util.find_spec(package)]
    if not_found:
        hint = Hint(
            "none",
            "The app's dependencies have changed",
            f"The following packages were not found: {COL_ERROR}{', '.join(not_found)}{COL_DEFAULT}\n"
            "You may have updated the app from the repository without updating its dependencies.\n"
            "Install the missing packages or update your virtual environment by re-running:\n"
            f"`pip install -r {Tts.get_type().value.requirements_file_name}`."
        )
        AppUtil.print_hint(hint)
        exit(1)

    # Show other one-time startup messages (which are not blockers)
    prefs = Prefs.load()

    if not does_import_test_pass("tkinter"): # To test for tkinter functionality, must do concrete import
        AppUtil.show_hint_if_necessary(prefs, HINT_TKINTER, and_prompt=True)

    if not is_long_path_enabled():
        AppUtil.show_hint_if_necessary(prefs, HINT_LONG_PATHS, and_prompt=True)

    if Tts.get_type() == TtsModelInfos.OUTE:
        AppUtil.show_hint_if_necessary(prefs, HINT_OUTE_CONFIG, and_prompt=True)
    elif Tts.get_type() == TtsModelInfos.INDEXTTS2:
        AppUtil.show_hint_if_necessary(prefs, HINT_INDEXTTS2, and_prompt=True)

    # Start
    printt()
    _ = App()

# ---

NEW_PACKAGES = ["faster_whisper", "audiotsm", "readchar", "psutil", "num2words"]
