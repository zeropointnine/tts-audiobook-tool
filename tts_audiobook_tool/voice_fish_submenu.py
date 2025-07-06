from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceFishSubmenu:

    @staticmethod
    def submenu(state: State) -> None:
        """
        """
        while True:
            VoiceFishSubmenu._print(state)
            hotkey = ask_hotkey()
            should_exit = VoiceFishSubmenu._handle_hotkey(state, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(state: State) -> None:

        print_heading(f"Voice clone and options")

        s = f"{make_hotkey_string("1")} Set voice clone audio clip"
        s += f" {COL_DIM}(currently: {COL_ACCENT}{state.project.get_voice_label()}{COL_DIM})"
        printt(s)

        s = f"{make_hotkey_string("2")} Clear voice clone"
        printt(s)

        temperature = state.project.fish_temperature
        s = "default" if temperature == -1 else str(temperature)
        printt(f"{make_hotkey_string("3")} Temperature {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")

        printt()

    @staticmethod
    def _handle_hotkey(state: State, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceFishSubmenu.ask_voice_file(state)
                return False
            case "2":
                state.project.clear_fish_voice_and_save()
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
                        state.project.fish_temperature = value
                        state.project.save()
                except:
                    ask_error("Bad value")
                    return False
            # case "3":
            #     value = ask("Enter value for exaggeration (0.25-2.0): ")
            #     if not value:
            #         return False
            #     try:
            #         value = float(value)
            #         if not (0.25 <= value <= 2.0):
            #             ask_error("Out of range")
            #         else:
            #             state.project.chatterbox_exaggeration = value
            #             state.project.save()
            #         return False
            #     except:
            #         ask_error("Bad value")
            #         return False
            # case "4":
            #     value = ask("Enter value for cfg/pace (0.2-1.0): ")
            #     if not value:
            #         return False
            #     try:
            #         value = float(value)
            #         if not (0.2 <= value <= 1.0):
            #             ask_error("Out of range")
            #         else:
            #             state.project.chatterbox_cfg = value
            #             state.project.save()
            #     except:
            #         ask_error("Bad value")
            #     return False
            case _:
                return True

        return False

    @staticmethod
    def ask_voice_file(state: State):

        path = VoiceSubmenuShared.ask_voice_file(state.project.dir_path)
        if not path:
            return

        # Load sound and transcribe
        sound = SoundFileUtil.load(path)
        if isinstance(sound, str):
            ask_continue(sound)
            return
        whisper_data = SoundUtil.transcribe(sound)
        if isinstance(whisper_data, str):
            ask_continue(whisper_data)
            return
        text = TranscribeUtil.get_whisper_data_text(whisper_data)

        err = state.project.set_fish_voice_and_save(path, text)
        if err:
            ask_error(err)
            return

        if MENU_CLEARS_SCREEN:
            ask_continue("Saved")
