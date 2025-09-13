from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceHiggsSubmenu:

    @staticmethod
    def submenu(project: Project) -> None:

        while True:
            VoiceHiggsSubmenu._print(project)
            hotkey = ask_hotkey()
            should_exit = VoiceHiggsSubmenu._handle_hotkey(project, hotkey)
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

        s = VoiceSubmenuShared.make_parameter_value_string(project.higgs_temperature, HIGGS_DEFAULT_TEMPERATURE, 1)
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('3')} Temperature {s}")

        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsModelInfos.HIGGS)
                return False
            case "2":
                project.clear_voice_and_save(TtsModelInfos.HIGGS)
                printt_set("Cleared")
                return False
            case "3":
                # TODO not sure about upper bound here (just taking a guess)
                #      2.0 is actually way too high in practice
                #      1.0 is perfectly fine on a gen-by-gen basis
                value = ask(f"Enter temperature (0.0 < value < 2.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 < value < 2.0):
                        ask_error("Out of range")
                    else:
                        project.higgs_temperature = value
                        project.save()
                        printt_set(f"Temperature set: {project.higgs_temperature}")
                except:
                    ask_error("Bad value")
                    return False
            case _:
                return True

        return False
