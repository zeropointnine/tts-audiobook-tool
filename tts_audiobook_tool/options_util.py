from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *


class OptionsUtil:

    @staticmethod
    def options_submenu(state: State) -> None:

        handled = True

        while handled:

            handled = False

            print_heading("Options:")
            printt(f"{make_hotkey_string("1")} Normalize audio after generate (currently: {state.prefs.should_normalize})")
            printt(f"{make_hotkey_string("2")} Play audio after each segment is generated (currently: {state.prefs.play_on_generate})")
            printt(f"{make_hotkey_string("3")} Transcode FLAC to MP4, preserving tts-audiobook-tool metadata")
            printt(f"{make_hotkey_string("4")} Add tts-audiobook-tool metadata to a pre-existing audiobook")

            if Shared.is_oute():
                s = "default" if state.project.oute_temperature == -1 else str(state.project.oute_temperature)
                printt(f"{make_hotkey_string("5")} Oute temperature (currently: {s})")
            elif Shared.is_chatterbox():
                s = "default" if state.project.chatterbox_temperature == -1 else str(state.project.chatterbox_temperature)
                printt(f"{make_hotkey_string("5")} Chatterbox temperature (currently: {s})")
                s = "default" if state.project.chatterbox_exaggeration == -1 else str(state.project.chatterbox_exaggeration)
                printt(f"{make_hotkey_string("6")} Chatterbox exaggeration (currently: {s})")
                s = "default" if state.project.chatterbox_cfg == -1 else str(state.project.chatterbox_cfg)
                printt(f"{make_hotkey_string("7")} Chatterbox cfg/pace (currently: {s})")
            printt()
            hotkey = ask_hotkey()

            match hotkey:
                case "1":
                    handled = True
                    state.prefs.should_normalize = not state.prefs.should_normalize
                    printt(f"Normalize audio after generate set to: {state.prefs.should_normalize}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()
                case "2":
                    handled = True
                    state.prefs.play_on_generate = not state.prefs.play_on_generate
                    printt(f"Play audio after each segment is generated set to: {state.prefs.play_on_generate}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()
                case "3":
                    handled = True
                    TranscodeUtil.ask_transcode(state)
                case "4":
                    handled = True
                    SttFlow.ask_make()

            if Shared.is_oute():
                match hotkey:
                    case "5":
                        handled = True
                        value = ask(f"Enter temperature (0.0 < value <= 2.0): ")
                        if not value:
                            return
                        try:
                            value = float(value)
                            if not (0.0 < value <= 2.0):
                                printt("Out of range", "error")
                            else:
                                state.project.oute_temperature = value
                                state.project.save()
                        except:
                            printt("Bad value", "error")

            elif Shared.is_chatterbox():
                match hotkey:
                    case "5":
                        handled = True
                        value = ask(f"Enter temperature (0.0 < value <= 5.0): ")
                        if not value:
                            return
                        try:
                            value = float(value)
                            if not (0.0 < value <= 5.0):
                                printt("Out of range", "error")
                            else:
                                state.project.chatterbox_temperature = value
                                state.project.save()
                        except:
                            printt("Bad value", "error")
                    case "6":
                        handled = True
                        value = ask("Enter value for exaggeration (0.25-2.0): ")
                        if not value:
                            return
                        try:
                            value = float(value)
                            if not (0.25 <= value <= 2.0):
                                printt("Out of range", "error")
                            else:
                                state.project.chatterbox_exaggeration = value
                                state.project.save()
                        except:
                            printt("Bad value", "error")
                    case "7":
                        handled = True
                        value = ask("Enter value for cfg/pace (0.2-1.0): ")
                        if not value:
                            return
                        try:
                            value = float(value)
                            if not (0.2 <= value <= 1.0):
                                printt("Out of range", "error")
                            else:
                                state.project.chatterbox_cfg = value
                                state.project.save()
                        except:
                            printt("Bad value", "error")
