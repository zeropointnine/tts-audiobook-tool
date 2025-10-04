from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model import FishProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceFishSubmenu:

    @staticmethod
    def submenu(project: Project) -> None:
        """
        """
        while True:
            VoiceFishSubmenu._print(project)
            hotkey = ask_hotkey()
            should_exit = VoiceFishSubmenu._handle_hotkey(project, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(project: Project) -> None:

        print_heading(f"Voice clone and model settings")

        label = make_currently_string(project.get_voice_label())
        s = f"{make_hotkey_string('1')} Select voice clone sample {label}"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.fish_temperature, FishProtocol.DEFAULT_TEMPERATURE, 1)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('3')} Temperature {s}")
        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsModelInfos.FISH)
                return False
            case "2":
                project.clear_voice_and_save(TtsModelInfos.FISH)
                printt_set("Cleared")
                printt()
                return False
            case "3":
                value = ask(f"Enter temperature (0.0 < value < 2.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 < value < 2.0):
                        ask_error("Out of range")
                    else:
                        project.fish_temperature = value
                        project.save()
                except:
                    ask_error("Bad value")
                    return False
            case _:
                return True

        return False
