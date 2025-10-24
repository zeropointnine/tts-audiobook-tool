from pathlib import Path

from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model import OuteProtocol
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
            should_exit = VoiceOuteSubmenu._handle_hotkey(state.project, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(state: State) -> None:

        print_heading(f"Voice clone and model settings")

        printt(f"{make_hotkey_string('1')} Set voice clone using audio clip (15s or less)")
        printt(f"{make_hotkey_string('2')} Set voice clone using Oute json file")
        printt(f"{make_hotkey_string('3')} Set voice clone to Oute default voice")

        s = VoiceSubmenuShared.make_parameter_value_string(
            state.project.oute_temperature, OuteProtocol.DEFAULT_TEMPERATURE, 1
        )
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('4')} Temperature {s}")
        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceOuteSubmenu.ask_create_oute_voice(project)
                return False
            case "2":
                VoiceOuteSubmenu.ask_load_oute_json(project)
                return False
            case "3":
                result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
                if isinstance(result, str):
                    ask_error(result)
                    return False
                project.set_oute_voice_and_save(result, "default")
                printt_set("Voice clone set.")
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
                        project.oute_temperature = value
                        project.save()
                except:
                    ask_error("Bad value")
                return False

            case _:
                return True

    @staticmethod
    def ask_create_oute_voice(project: Project) -> None:

        from tts_audiobook_tool.app_util import AppUtil

        path = VoiceSubmenuShared.ask_voice_file(project.dir_path, Tts.get_type())
        if not path:
            return

        # Outte is about to create its own instance of whisper, so better clear ours first
        Stt.clear_stt_model()
        AppUtil.gc_ram_vram()

        result = Tts.get_oute().create_speaker(path)

        # Clear lingering oute-created whisper instance
        AppUtil.gc_ram_vram()

        if isinstance(result, str):
            error = result
            ask_error(f"Error creating voice: {error}")
            return

        voice_dict = result
        project.set_oute_voice_and_save(voice_dict, Path(path).stem)

        printt()
        printt_set("Voice clone set.")


    @staticmethod
    def ask_load_oute_json(project: Project):

        path = ask_file_path(
            "Enter file path of voice json file: ",
            "Select Oute voice json file",
            [("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=project.dir_path
        )
        if not path:
            return

        result = OuteUtil.load_oute_voice_json(path)
        if isinstance(result, str):
            ask_error(result)
            return

        project.set_oute_voice_and_save(result, Path(path).stem)
        printt_set("Voice clone set.")
