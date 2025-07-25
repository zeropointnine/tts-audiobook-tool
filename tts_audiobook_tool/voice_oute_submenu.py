from pathlib import Path

from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

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

        print_heading(f"Voice clone and options")

        printt(f"{make_hotkey_string('1')} Set voice clone using audio clip (15s or less)")
        printt(f"{make_hotkey_string('2')} Set voice clone using Oute json file")
        printt(f"{make_hotkey_string('3')} Set voice clone to Oute default voice")
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
                VoiceOuteSubmenu.ask_load_oute_json(state)
                return False
            case "3":
                result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
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

        path = VoiceSubmenuShared.ask_voice_file(state.project.dir_path)
        if not path:
            return

        # Outte is about to load its own instance of whisper, so better clear ours first
        Tts.clear_whisper()
        AppUtil.gc_ram_vram()

        interface = Tts.get_oute()
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
    def ask_load_oute_json(state: State):

        path = ask_file_path(
            "Enter file path of voice json file: ",
            "Select Oute voice json file",
            [("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=state.project.dir_path
        )
        if not path:
            return

        result = OuteUtil.load_oute_voice_json(path)
        if isinstance(result, str):
            ask_error(result)
            return

        state.project.set_oute_voice_and_save(result, Path(path).stem)
