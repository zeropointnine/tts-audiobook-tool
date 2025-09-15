from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model import ChatterboxProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceChatterboxSubmenu:

    @staticmethod
    def submenu(project: Project) -> None:
        """
        """
        while True:
            VoiceChatterboxSubmenu._print(project)
            hotkey = ask_hotkey()
            should_exit = VoiceChatterboxSubmenu._handle_hotkey(project, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(project: Project) -> None:

        print_heading(f"Voice clone and model settings")

        s = make_currently_string(project.get_voice_label())
        s = f"{make_hotkey_string('1')} Select voice clone sample {s}"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.chatterbox_temperature, ChatterboxProtocol.DEFAULT_TEMPERATURE, 1)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('3')} Temperature {s}")

        s = VoiceSubmenuShared.make_parameter_value_string(project.chatterbox_exaggeration, ChatterboxProtocol.DEFAULT_EXAGGERATION, 1)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('4')} Exaggeration {s}")

        s = VoiceSubmenuShared.make_parameter_value_string(project.chatterbox_cfg, ChatterboxProtocol.DEFAULT_CFG, 1)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('5')} Cfg/pace {s}")
        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsModelInfos.CHATTERBOX)
                return False
            case "2":
                project.clear_voice_and_save(TtsModelInfos.CHATTERBOX)
                printt("Cleared")
                printt()
                return False
            case "3":
                value = ask(f"Enter temperature (0.0 < value <= 3.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 < value <= 3.0):
                        ask_error("Out of range")
                    else:
                        project.chatterbox_temperature = value
                        project.save()
                except:
                    ask_error("Bad value")
                    return False
            case "4":
                value = ask("Enter value for exaggeration (0.25-2.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.25 <= value <= 2.0):
                        ask_error("Out of range")
                    else:
                        project.chatterbox_exaggeration = value
                        project.save()
                    return False
                except:
                    ask_error("Bad value")
                    return False
            case "5":
                value = ask("Enter value for cfg/pace (0.2-1.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.2 <= value <= 1.0):
                        ask_error("Out of range")
                    else:
                        project.chatterbox_cfg = value
                        project.save()
                except:
                    ask_error("Bad value")
                return False
            case _:
                return True

        return False

