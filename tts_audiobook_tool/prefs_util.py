import json
from pathlib import Path
import time

from tts_audiobook_tool.l import L
from tts_audiobook_tool.state import State

from .util import *
from .constants import *

class PrefsUtil:
    """
    User settings that persist.
    temperature
    play_on_generate
    """

    @staticmethod
    def load_from_disk(state: State) -> str:
        """
        Loads pref values from disk and updates state object
        Returns user-facing error message on fail
        """

        # Set values to defaults first
        state.temperature = DEFAULT_TEMPERATURE
        state.play_on_generate = False

        if not os.path.exists(PrefsUtil.get_file_path()):
            PrefsUtil.save_to_disk(state)
            return ""

        try:
            with open(PrefsUtil.get_file_path(), 'r', encoding='utf-8') as f:
                dic = json.load(f)
                if not isinstance(dic, dict):
                    PrefsUtil.save_to_disk(state)
                    return "Prefs file error: Bad data"
        except Exception as e:
            PrefsUtil.save_to_disk(state)
            return f"Prefs file error: {e}"

        temperature = dic.get("temperature", DEFAULT_TEMPERATURE)
        if not isinstance(temperature, float) or not (0 < temperature <= 2.0):
            temperature = DEFAULT_TEMPERATURE
            PrefsUtil.save_to_disk(state)
        state.temperature = temperature

        play_on_generate = dic.get("play_on_generate", False)
        if not isinstance(play_on_generate, bool):
            play_on_generate = False
            PrefsUtil.save_to_disk(state)
        state.play_on_generate = play_on_generate

        return ""

    @staticmethod
    def save_to_disk(state: State) -> None:
        """
        Saves to disk using state object
        Fails silently # TODO revisit?
        """
        dic = {
            "temperature": state.temperature,
            "play_on_generate": state.play_on_generate
        }
        try:
            with open(PrefsUtil.get_file_path(), 'w', encoding='utf-8') as f:
                json.dump(dic, f, indent=4)
        except Exception as e:
            L.w(f"{e}")

    @staticmethod
    def get_file_path() -> str:
        dir = Path.home()
        return os.path.join(dir, PREFS_FILE_NAME)
