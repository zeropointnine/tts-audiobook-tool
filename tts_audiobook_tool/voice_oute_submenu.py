import json
import os
from pathlib import Path

from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class VoiceOuteSubmenu:

    @staticmethod
    def submenu(state: State) -> None:
        """
        """
        while True:
            VoiceOuteSubmenu._print(state)
            hotkey = ask_hotkey()
            should_exit = VoiceOuteSubmenu._handle_hotkey(state, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(state: State) -> None:

        print_heading(f"Voice clone and options ({Shared.get_model_label()})")
        printt(f"{make_hotkey_string('1')} Set voice using reference WAV file (15s or less)")
        printt(f"{make_hotkey_string('2')} Set voice using Oute voice json file")
        printt(f"{make_hotkey_string('3')} Set Oute default voice")
        temp = state.project.oute_temperature
        s = "default" if temp == -1 else str(temp)
        printt(f"{make_hotkey_string("5")} Oute temperature (currently: {s})")
        printt()

    @staticmethod
    def _handle_hotkey(state: State, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceOuteSubmenu.ask_create_oute_voice(state)
                return False
            case "2":
                VoiceOuteSubmenu.ask_load_oute_voice(state)
                return False
            case "3":
                result = OuteUtil.load_oute_voice_json(DEFAULT_VOICE_JSON_FILE_PATH)
                if isinstance(result, str):
                    printt(result, "error")
                    return False
                state.project.set_oute_voice_and_save(result, "default")
                return False
            case "4":
                value = ask(f"Enter temperature (0.0 < value <= 2.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 < value <= 2.0):
                        printt("Out of range", "error")
                    else:
                        state.project.oute_temperature = value
                        state.project.save()
                except:
                    printt("Bad value", "error")
                return False

            case _:
                return True

    @staticmethod
    def ask_create_oute_voice(state: State) -> None:

        path = ask_path("Enter file path of source audio (up to 15s) for voice clone:\n")
        if not path:
            return
        if not os.path.exists(path):
            ask_continue(f"File not found: {path}")
            return

        interface = Shared.get_oute()
        try:
            voice_json = interface.create_speaker(path)
            printt()
        except Exception as e:
            printt(f"Error creating voice: {e}", "error")
            return

        state.project.set_oute_voice_and_save(voice_json, Path(path).stem)

        # Outte created a whisper instance, which will stick around
        # (in addition to app's own whisper instance)
        # if we don't explicitly flush stuffs
        from tts_audiobook_tool.app_util import AppUtil
        AppUtil.gc_ram_vram()

    @staticmethod
    def ask_load_oute_voice(state: State):

        path = ask("Enter file path of voice json file:\n")
        if not path:
            return

        result = OuteUtil.load_oute_voice_json(path)
        if isinstance(result, str):
            printt(result, "error")
            return

        state.project.set_oute_voice_and_save(result, Path(path).stem)

