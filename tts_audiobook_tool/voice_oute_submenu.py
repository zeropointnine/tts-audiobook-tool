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

        s = f"{COL_DIM}(currently: {COL_ACCENT}{state.project.get_voice_label()}{COL_DIM})"
        print_heading(f"Voice clone and model options {s}")

        printt(f"{make_hotkey_string('1')} Set Oute voice using reference WAV file (15s or less)")
        printt(f"{make_hotkey_string('2')} Set Oute voice using Oute voice json file")
        printt(f"{make_hotkey_string('3')} Set Oute default voice")
        temp = state.project.oute_temperature
        s = "default" if temp == -1 else str(temp)
        printt(f"{make_hotkey_string("4")} Temperature (currently: {s})")
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
                    ask_error(result)
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
                        ask_error("Out of range")
                    else:
                        state.project.oute_temperature = value
                        state.project.save()
                except:
                    ask_error("Bad value")
                return False

            case _:
                return True

    @staticmethod
    def ask_create_oute_voice(state: State) -> None:

        from tts_audiobook_tool.app_util import AppUtil

        path = ask_path("Enter file path of source audio (up to 15s) for voice clone:\n")
        if not path:
            return
        if not os.path.exists(path):
            ask_continue(f"File not found: {path}")
            return

        # Outte is about to load its own instance of whisper, so better clear ours first
        Shared.clear_whisper()
        AppUtil.gc_ram_vram()

        interface = Shared.get_oute()
        try:
            voice_json = interface.create_speaker(path)
            printt()
        except Exception as e:
            ask_error(f"Error creating voice: {e}")
            return

        state.project.set_oute_voice_and_save(voice_json, Path(path).stem)

        # Outte created a whisper instance so force its eviction
        AppUtil.gc_ram_vram()

    @staticmethod
    def ask_load_oute_voice(state: State):

        path = ask("Enter file path of voice json file:\n")
        if not path:
            return

        result = OuteUtil.load_oute_voice_json(path)
        if isinstance(result, str):
            ask_error(result)
            return

        state.project.set_oute_voice_and_save(result, Path(path).stem)
