from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *

class OptionsSubmenu:

    @staticmethod
    def submenu(state: State) -> None:

        while True:

            vram_bytes = get_torch_allocated_vram()
            if vram_bytes == -1:
                vram_label = ""
            else:
                vram_label = f"{COL_DIM}(currently allocated VRAM: {make_gb_string(vram_bytes)})"

            print_heading("Options:")
            value = make_currently_string(state.project.stt_variant.value)
            printt(f"{make_hotkey_string('1')} Whisper transcription model {value}")
            value = make_currently_string(str(state.prefs.section_sound_effect))
            printt(f"{make_hotkey_string('2')} Use page turn sound effect at section breaks {value}")
            printt(f"{make_hotkey_string('3')} Attempt to unload models {vram_label}")
            printt(f"{make_hotkey_string('4')} Reset contextual hints")
            printt()

            hotkey = ask_hotkey()
            match hotkey:
                case "1":
                    OptionsSubmenu.ask_transcription_model(state)
                case "2":
                    AppUtil.show_hint_if_necessary(state.prefs, HINT_SECTION_SOUND_EFFECT)
                    state.prefs.section_sound_effect = not state.prefs.section_sound_effect
                    printt_set(f"Section break sound effect has been toggled to: {state.prefs.section_sound_effect}")
                case "3":
                    Tts.clear_all_models()
                    printt_set("Finished")
                case "4":
                    state.prefs.reset_hints()
                    printt("One-time contextual hints have been reset.\nThey will now appear again when relevant.\n")
                    ask_continue()
                case _:
                    break

    @staticmethod
    def ask_transcription_model(state: State) -> None:

        while True:

            print_heading(f"Select Whisper transcription model {make_currently_string(state.project.stt_variant.value)}")
            printt()
            AppUtil.show_hint_if_necessary(state.prefs, HINT_TRANSCRIPTION)
            printt(f"{make_hotkey_string('1')} large-v3 {COL_DIM}(best accuracy, default)")
            printt(f"{make_hotkey_string('2')} large-v3-turbo {COL_DIM}(slightly less memory, faster)")
            printt()

            hotkey = ask_hotkey()
            if not hotkey:
                break

            value = None
            match hotkey:
                case "1":
                    value = SttVariant.LARGE_V3
                case "2":
                    value = SttVariant.LARGE_V3_TURBO

            if value:
                state.project.stt_variant = value
                state.project.save()
                Stt.set_variant_using_project(state.project)
                printt_set(f"Whisper transcription model set to: {state.project.stt_variant.value}")
                break
