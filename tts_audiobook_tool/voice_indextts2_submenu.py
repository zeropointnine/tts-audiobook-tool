from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import IndexTts2Protocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceIndexTts2Submenu:

    @staticmethod
    def submenu(project: Project) -> None:
        """
        """
        while True:
            VoiceIndexTts2Submenu._print(project)
            hotkey = ask_hotkey()
            should_exit = VoiceIndexTts2Submenu._handle_hotkey(project, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(project: Project) -> None:

        print_heading(f"Voice clone and model settings")

        label = make_currently_string(project.get_voice_label())
        required = f"{COL_DIM}(required){COL_DEFAULT} " if not project.indextts2_voice_file_name else ""
        s = f"{make_hotkey_string('1')} Select voice clone sample {required}{label}"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone sample"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_temperature, IndexTts2Protocol.DEFAULT_TEMPERATURE, 2)
        printt(f"{make_hotkey_string('3')} Temperature {make_currently_string(s)}")

        label = make_currently_string(project.get_voice_label(is_secondary=True))
        optional = f"{COL_DIM}(optional){COL_DEFAULT} " if not project.indextts2_emo_voice_file_name else ""
        s = f"{make_hotkey_string('4')} Select emotion voice sample {optional}{label}"
        printt(s)

        s = f"{make_hotkey_string('5')} Clear emotion voice sample"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_emo_voice_alpha, IndexTts2Protocol.DEFAULT_EMO_VOICE_ALPHA, 2)
        printt(f"{make_hotkey_string('6')} Emotion voice sample strength {make_currently_string(s)}")

        s = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_use_fp16, IndexTts2Protocol.DEFAULT_USE_FP16)
        printt(f"{make_hotkey_string('7')} Toggle FP16 (smaller memory footprint) {make_currently_string(s)}")

        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsModelInfos.INDEXTTS2)
                return False
            case "2":
                project.clear_voice_and_save(TtsModelInfos.INDEXTTS2)
                printt_set("Cleared")
                return False
            case "3":
                value = ask(f"Enter temperature (0.1 <= value <= 2.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.1 <= value <= 2.0):
                        ask_error("Out of range")
                    else:
                        project.indextts2_temperature = value
                        project.save()
                except:
                    ask_error("Bad value")
                    return False
            case "4":
                # TODO: disallow emo voice file == voice file (bc is default behavior anyway)
                VoiceSubmenuShared.ask_and_set_voice_file(project, TtsModelInfos.INDEXTTS2, is_secondary=True)
                return False
            case "5":
                project.clear_voice_and_save(TtsModelInfos.INDEXTTS2, is_secondary=True)
                printt_set("Cleared")
                return False
            case "6":
                value = ask(f"Enter emotion voice sample strength (0.0 <= value <= 1.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 <= value <= 1.0):
                        ask_error("Out of range")
                    else:
                        project.indextts2_emo_voice_alpha = value
                        project.save()
                except:
                    ask_error("Bad value")
                    return False
            case "7":
                project.indextts2_use_fp16 = not project.indextts2_use_fp16
                project.save()
                Tts.set_model_params_using_project(project)
                printt_set(f"FP16 set to: {project.indextts2_use_fp16}")

            case _:
                return True

        return False
