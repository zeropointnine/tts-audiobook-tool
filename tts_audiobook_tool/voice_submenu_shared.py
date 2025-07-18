import os

from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class VoiceSubmenuShared:

    @staticmethod
    def ask_voice_file(default_dir_path) -> str:
        """
        Asks for voice file path.
        Shows error prompt if necessary.
        Returns path or empty string.
        """

        ui = Tts.get_type().value.ui
        valid_suffixes = ui["voice_path_suffixes"]

        path = ask_file_path(
             console_message=ui.get("voice_path_console", ""),
             requestor_title=ui.get("voice_path_requestor", ""),
             initialdir=default_dir_path
        )
        if not path:
            return ""

        if not os.path.exists(path):
            ask_error(f"File not found: {path}")
            return ""

        suffix = Path(path).suffix
        if not suffix in valid_suffixes:
            ask_error(f"Invalid suffix: {suffix}. Must be {valid_suffixes}")
            return ""

        err = SoundFileUtil.is_valid_sound_file(path)
        if err:
            ask_error(err)
            return ""

        return path