from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceVibeVoiceSubmenu:

    @staticmethod
    def submenu(project: Project) -> None:

        while True:
            VoiceVibeVoiceSubmenu._print(project)
            hotkey = ask_hotkey()
            should_exit = VoiceVibeVoiceSubmenu._handle_hotkey(project, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(project: Project) -> None:

        print_heading(f"Voice clone and options")

        label = make_currently_string(project.get_voice_label())
        s = f"{make_hotkey_string('1')} Select voice clone sample {label}"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.vibevoice_cfg, VIBEVOICE_DEFAULT_CFG, 1)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('3')} CFG scale {s}")

        s = VoiceSubmenuShared.make_parameter_value_string(project.vibevoice_steps, VIBEVOICE_DEFAULT_NUM_STEPS, 0)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('4')} Steps {s}")
        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:
        """ Returns true if should exit submenu """

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsModelInfos.VIBEVOICE)
                return False
            case "2":
                project.clear_voice_and_save(TtsModelInfos.VIBEVOICE)
                printt("Cleared")
                printt()
                return False
            case "3":
                # Sane legal range (IMO)
                value = ask(f"Enter cfg (1.3 <= value <= 7.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (1.3 <= value <= 7.0):
                        ask_error("Out of range")
                    else:
                        project.vibevoice_cfg = value
                        project.save()
                except:
                    ask_error("Bad value")
                return False
            case "4":
                # Sane legal range (IMO)
                value = ask(f"Enter num steps (1 <= value <= 30): ")
                if not value:
                    return False
                try:
                    value = int(value)
                    if not (1 <= value <= 30.0):
                        ask_error("Out of range")
                    else:
                        project.vibevoice_steps = value
                        project.save()
                except:
                    ask_error("Bad value")
                return False

            case _:
                return True
