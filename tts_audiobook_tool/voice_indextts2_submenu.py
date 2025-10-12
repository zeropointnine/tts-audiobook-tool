from tts_audiobook_tool.app_util import AppUtil
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

        if project.indextts2_voice_file_name:
            current = make_currently_string(project.get_voice_label())
        else:
            current = f"{COL_ERROR}(required){COL_DEFAULT}"
        s = f"{make_hotkey_string('1')} Select voice clone sample {current}"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone sample"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_temperature, IndexTts2Protocol.DEFAULT_TEMPERATURE, 2)
        printt(f"{make_hotkey_string('3')} Temperature {make_currently_string(s)}")

        if project.indextts2_emo_voice_file_name:
            current = make_currently_string(project.get_voice_label(is_secondary=True))
        else:
            current = f"{COL_DIM}(optional){COL_DEFAULT}"
        s = f"{make_hotkey_string('4')} Select emotion voice sample {current}"
        printt(s)

        s = f"{make_hotkey_string('5')} Clear emotion voice sample"
        printt(s)

        if project.indextts2_emo_vector:
            current = make_currently_string(project.emo_vector_to_string())
        else:
            current = f"{COL_DIM}(optional){COL_DEFAULT}"
        s = f"{make_hotkey_string('6')} Emotion vector {current}"
        printt(s)

        s = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_emo_alpha, IndexTts2Protocol.DEFAULT_EMO_VOICE_ALPHA, 2)
        printt(f"{make_hotkey_string('7')} Emotion alpha (strength) {make_currently_string(s)}")

        s = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_use_fp16, IndexTts2Protocol.DEFAULT_USE_FP16)
        printt(f"{make_hotkey_string('8')} Toggle FP16 (smaller memory footprint) {make_currently_string(s)}")

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
                VoiceSubmenuShared.ask_and_set_voice_file(
                    project,
                    TtsModelInfos.INDEXTTS2,
                    is_secondary=True,
                    message_override="Enter emotion reference audio clip file path:"
                )
                return False
            case "5":
                project.clear_voice_and_save(TtsModelInfos.INDEXTTS2, is_secondary=True)
                printt_set("Cleared")
                return False
            case "6":
                VoiceIndexTts2Submenu.ask_vector(project)
                pass
            case "7":
                printt(f"Enter emotion alpha (0.0 <= value <= 1.0): ")
                value = ask()
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 <= value <= 1.0):
                        ask_error("Out of range")
                    else:
                        project.indextts2_emo_alpha = value
                        project.save()
                except:
                    ask_error("Bad value")
                    return False
            case "8":
                project.indextts2_use_fp16 = not project.indextts2_use_fp16
                project.save()
                Tts.set_model_params_using_project(project)
                printt_set(f"FP16 set to: {project.indextts2_use_fp16}")

            case _:
                return True

        return False

    @staticmethod
    def ask_vector(project: Project) -> None:

        printt("Enter emotion vector:")
        printt()
        s = "This is a list of eight numbers between 0-1 corresponding to:\n"
        s += "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm\n"
        s += f'{COL_DIM}Eg: "0, 0.8, 0, 0, 0.2, 0, 0, 0" = very angry, slightly disgusted{COL_DEFAULT}'
        printt(s)
        printt()
        inp = ask("")
        if not inp:
            return
        result = Project.parse_emo_vector_string(inp)
        if isinstance(result, str):
            ask_error(result)
        else:
            project.indextts2_emo_vector = result
            project.save()
            printt_set(f"Emotion vector saved: {project.emo_vector_to_string()}")
