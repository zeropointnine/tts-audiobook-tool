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
            printt(f"{make_hotkey_string("3")} Create tts-audiobook-tool metadata for a pre-existing audiobook")
            printt(f"{make_hotkey_string("4")} Transcode FLAC to MP4, preserving tts-audiobook-tool metadata")

            if Shared.is_oute():
                oute_temp = state.project.oute_temperature
                s = "default" if oute_temp == -1 else str(oute_temp)
                printt(f"{make_hotkey_string("5")} Oute temperature (currently: {s})")
            elif Shared.is_chatterbox():
                cb_temp = state.project.chatterbox_temperature
                s = "default" if cb_temp == -1 else str(cb_temp)
                printt(f"{make_hotkey_string("5")} Chatterbox temperature (currently: {s})")
                cb_ex = state.project.chatterbox_exaggeration
                s = "default" if cb_ex == -1 else str(cb_ex)
                printt(f"{make_hotkey_string("6")} Chatterbox exaggeration (currently: {s})")
                cb_cfg = state.project.chatterbox_cfg
                s = "default" if cb_cfg == -1 else str(cb_cfg)
                printt(f"{make_hotkey_string("7")} Chatterbox cfg/pace (currently: {s})")
            printt()
            hotkey = ask_hotkey()

            match hotkey:
                case "1":
                    handled = True
                    state.prefs.should_normalize = not state.prefs.should_normalize
                    printt(f"Apply dynamic loudness normalization to audio upon generate: {state.prefs.should_normalize}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()
                case "2":
                    handled = True
                    state.prefs.play_on_generate = not state.prefs.play_on_generate
                    printt(f"Play audio after each segment is generated: {state.prefs.play_on_generate}")
                    printt()
                    if MENU_CLEARS_SCREEN:
                        ask_continue()
                case "3":
                    handled = True
                    SttFlow.ask_make(state.prefs)
                case "4":
                    handled = True
                    TranscodeUtil.ask_transcode(state)

            if Shared.is_oute():
                match hotkey:
                    case "5":
                        handled = True
                        oute_temp = ask(f"Enter temperature (0.0 < value <= 2.0): ")
                        if not oute_temp:
                            return
                        try:
                            oute_temp = float(oute_temp)
                            if not (0.0 < oute_temp <= 2.0):
                                printt("Out of range", "error")
                            else:
                                state.project.oute_temperature = oute_temp
                                state.project.save()
                        except:
                            printt("Bad value", "error")

            elif Shared.is_chatterbox():
                match hotkey:
                    case "5":
                        handled = True
                        oute_temp = ask(f"Enter temperature (0.0 < value <= 5.0): ")
                        if not oute_temp:
                            return
                        try:
                            oute_temp = float(oute_temp)
                            if not (0.0 < oute_temp <= 5.0):
                                printt("Out of range", "error")
                            else:
                                state.project.chatterbox_temperature = oute_temp
                                state.project.save()
                        except:
                            printt("Bad value", "error")
                    case "6":
                        handled = True
                        oute_temp = ask("Enter value for exaggeration (0.25-2.0): ")
                        if not oute_temp:
                            return
                        try:
                            oute_temp = float(oute_temp)
                            if not (0.25 <= oute_temp <= 2.0):
                                printt("Out of range", "error")
                            else:
                                state.project.chatterbox_exaggeration = oute_temp
                                state.project.save()
                        except:
                            printt("Bad value", "error")
                    case "7":
                        handled = True
                        oute_temp = ask("Enter value for cfg/pace (0.2-1.0): ")
                        if not oute_temp:
                            return
                        try:
                            oute_temp = float(oute_temp)
                            if not (0.2 <= oute_temp <= 1.0):
                                printt("Out of range", "error")
                            else:
                                state.project.chatterbox_cfg = oute_temp
                                state.project.save()
                        except:
                            printt("Bad value", "error")
