from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_info import TtsType
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

        s = f"{make_hotkey_string('1')} Select voice clone sample"
        s += f" {COL_DIM}(currently: {COL_ACCENT}{project.get_voice_label()}{COL_DIM})"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone"
        printt(s)

        cfg = DEFAULT_CFG_VIBEVOICE if project.vibevoice_cfg == -1 else project.vibevoice_cfg
        s = f"{cfg:.1f}"
        if cfg == DEFAULT_CFG_VIBEVOICE:
            s += " (default)"
        printt(f"{make_hotkey_string('3')} CFG scale {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")

        steps = DEFAULT_NUM_STEPS_VIBE_VOICE if project.vibevoice_steps == -1 else project.vibevoice_steps
        s = str(steps)
        if steps == DEFAULT_NUM_STEPS_VIBE_VOICE:
            s += " (default)"
        printt(f"{make_hotkey_string('4')} Steps {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")

        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:
        """ Returns true if should exit submenu """

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsType.VIBEVOICE)
                return False
            case "2":
                project.clear_voice_and_save(TtsType.VIBEVOICE)
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
