import os

from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class VoiceChatterboxSubmenu:

    @staticmethod
    def submenu(state: State) -> None:
        """
        """
        while True:
            VoiceChatterboxSubmenu._print(state)
            hotkey = ask_hotkey()
            should_exit = VoiceChatterboxSubmenu._handle_hotkey(state, hotkey)
            if should_exit:
                return

    @staticmethod
    def _print(state: State) -> None:

        s = f"{COL_DIM}(currently: {COL_ACCENT}{state.project.get_voice_label()}{COL_DIM})"
        print_heading(f"Voice clone and model options {s}")

        s = f"{make_hotkey_string('1')} Set Chatterbox voice "
        s += f"{COL_DIM}(currently: {COL_ACCENT}{state.project.get_voice_label()}{COL_DIM})"
        printt(s)
        cb_temp = state.project.chatterbox_temperature
        s = "default" if cb_temp == -1 else str(cb_temp)
        printt(f"{make_hotkey_string("2")} Temperature {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")
        cb_ex = state.project.chatterbox_exaggeration
        s = "default" if cb_ex == -1 else str(cb_ex)
        printt(f"{make_hotkey_string("3")} Exaggeration {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")
        cb_cfg = state.project.chatterbox_cfg
        s = "default" if cb_cfg == -1 else str(cb_cfg)
        printt(f"{make_hotkey_string("4")} Cfg/pace {COL_DIM}(currently: {COL_ACCENT}{s}{COL_DIM})")
        printt()

    @staticmethod
    def _handle_hotkey(state: State, hotkey: str) -> bool:

        match hotkey:
            case "1":
                VoiceChatterboxSubmenu.ask_voice_file(state)
                return False
            case "2":
                value = ask(f"Enter temperature (0.0 < value <= 5.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.0 < value <= 5.0):
                        ask_error("Out of range")
                    else:
                        state.project.chatterbox_temperature = value
                        state.project.save()
                except:
                    ask_error("Bad value")
                    return False
            case "3":
                value = ask("Enter value for exaggeration (0.25-2.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.25 <= value <= 2.0):
                        ask_error("Out of range")
                    else:
                        state.project.chatterbox_exaggeration = value
                        state.project.save()
                    return False
                except:
                    ask_error("Bad value")
                    return False
            case "4":
                value = ask("Enter value for cfg/pace (0.2-1.0): ")
                if not value:
                    return False
                try:
                    value = float(value)
                    if not (0.2 <= value <= 1.0):
                        ask_error("Out of range")
                    else:
                        state.project.chatterbox_cfg = value
                        state.project.save()
                except:
                    ask_error("Bad value")
                return False
            case _:
                return True

        return False

    @staticmethod
    def ask_voice_file(state: State):

        printt("Enter file path of source audio for voice clone:")
        path = ask_path()
        if not path:
            return

        if not os.path.exists(path):
            ask_continue(f"File not found: {path}")
            return

        err = SoundFileUtil.is_valid_sound_file(path)
        if err:
            ask_error(err)
            return

        err = state.project.set_chatterbox_voice_and_save(path)

        if err:
            ask_error(err)
        elif MENU_CLEARS_SCREEN:
                ask_continue("Saved.")
