from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.transcribe_util import TranscribeUtil
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

        s = f"{make_hotkey_string("1")} Set voice clone audio clip"
        s += f" {COL_DIM}(currently: {COL_ACCENT}{project.get_voice_label()}{COL_DIM})"
        printt(s)

        s = f"{make_hotkey_string("2")} Clear voice clone"
        printt(s)

        temperature = project.higgs_temperature
        s = f"default ({DEFAULT_TEMPERATURE_HIGGS})" if temperature == -1 else str(temperature)
        printt(f"{make_hotkey_string("3")} Temperature {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")

        printt()

    @staticmethod
    def _handle_hotkey(project: Project, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceHiggsSubmenu.ask_voice_file(project)
                return False
            case "2":
                project.clear_higgs_voice_and_save()
                printt("Cleared")
                printt()
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
                except:
                    ask_error("Bad value")
                    return False
            case _:
                return True

        return False

    @staticmethod
    def ask_voice_file(project: Project):

        path = VoiceSubmenuShared.ask_voice_file(project.dir_path)
        if not path:
            return

        # Load sound and transcribe
        sound = SoundFileUtil.load(path)
        if isinstance(sound, str):
            ask_continue(sound)
            return
        result = SoundUtil.transcribe(sound)
        if isinstance(result, str):
            ask_continue(result)
            return
        text = TranscribeUtil.get_whisper_data_text(result)

        err = project.set_higgs_voice_and_save(path, text)
        if err:
            ask_error(err)
            return

        if MENU_CLEARS_SCREEN:
            ask_continue("Saved")
