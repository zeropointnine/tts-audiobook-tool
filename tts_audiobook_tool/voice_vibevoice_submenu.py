from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import VibeVoiceProtocol
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

        print_heading(f"Voice clone and model settings")

        label = make_currently_string(project.get_voice_label())
        s = f"{make_hotkey_string('1')} Select voice clone sample {label}"
        printt(s)

        s = f"{make_hotkey_string('2')} Clear voice clone"
        printt(s)

        s = project.vibevoice_model_path if project.vibevoice_model_path else "none"
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('3')} VibeVoice custom model path {s}")

        s = VoiceSubmenuShared.make_parameter_value_string(
            project.vibevoice_cfg, VibeVoiceProtocol.DEFAULT_CFG, 1
        )
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('4')} CFG scale {s}")

        s = VoiceSubmenuShared.make_parameter_value_string(
            project.vibevoice_steps, VibeVoiceProtocol.DEFAULT_NUM_STEPS, 0
        )
        s = make_currently_string(s)
        printt(f"{make_hotkey_string('5')} Steps {s}")
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
                s = "Select custom local directory containing VibeVoice model (Hugging Face model repository format)"
                dir_path = ask_dir_path(s, s)
                if dir_path == project.vibevoice_model_path:
                    return False
                if dir_path and not os.path.exists(dir_path):
                    ask_continue("No such directory")
                    return False
                VoiceVibeVoiceSubmenu._apply_model_path_and_validate(project, dir_path)
                return False
            case "4":
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
            case "5":
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

    @staticmethod
    def _apply_model_path_and_validate(project: Project, path: str) -> None:

        project.vibevoice_model_path = path
        project.save()

        Tts.set_model_params_using_project(project)

        if not path:
            # No need to validate
            printt_set(f"Set to none; will use default model {VibeVoiceProtocol.DEFAULT_MODEL_NAME}")
            return

        # Validate by attempting to instantiate model with new settings

        # Model should have been cleared, but just in case:
        model = Tts.get_tts_model_if_exists()
        if model:
            Tts.clear_tts_model()

        try:
            _ = Tts.get_vibevoice()
            printt_set(f"\nCustom model path set: {path}")

        except (OSError, Exception) as e:
            # Revert change
            project.vibevoice_model_path = ""
            project.save()
            Tts.set_model_params_using_project(project)

            printt()
            printt("Contents at model path {path} appear to be invalid:")
            printt(make_error_string(e))
            printt()
            ask_continue()

