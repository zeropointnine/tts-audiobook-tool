from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_info import TtsType
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

        print_heading(f"Voice clone and options")

        s = f"{make_hotkey_string("1")} Set voice clone audio clip"
        s += f" {COL_DIM}(currently: {COL_ACCENT}{project.get_voice_label()}{COL_DIM})"
        printt(s)

        s = f"{make_hotkey_string("2")} Clear voice clone"
        printt(s)

        temperature = project.fish_temperature
        s = "default" if temperature == -1 else str(temperature)
        printt(f"{make_hotkey_string("3")} Temperature {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")

        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsType.FISH)
                return False
            case "2":
                project.clear_voice_and_save(TtsType.FISH)
                printt("Cleared")
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
