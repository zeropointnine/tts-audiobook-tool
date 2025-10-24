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

            result = AppUtil.get_nv_vram()
            if result is None:
                vram_label = ""
            else:
                used, total = result
                vram_label = f"{COL_DIM}(system VRAM: {make_gb_string(used)}/{make_gb_string(total)})"

            print_heading("Options:")
            value = make_currently_string(state.prefs.stt_variant.id)
            printt(f"{make_hotkey_string('1')} Whisper transcription model {value}")
            value = make_currently_string(str(state.prefs.use_section_sound_effect))
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
                    state.prefs.use_section_sound_effect = not state.prefs.use_section_sound_effect
                    printt_set(f"Section break sound effect has been toggled to: {state.prefs.use_section_sound_effect}")
                case "3":
                    result = Tts.clear_all_models()
                    if result:
                        message = f"VRAM usage before: {make_gb_string(result[0])}\n"
                        message += f"VRAM usage after: {make_gb_string(result[1])}"
                    else:
                        message = "OK"
                    printt_set(message)
                case "4":
                    state.prefs.reset_hints()
                    printt("One-time contextual hints have been reset.\nThey will now appear again when relevant.\n")
                    ask_continue()
                case _:
                    break

    @staticmethod
    def ask_transcription_model(state: State) -> None:

        while True:

            value = make_currently_string(state.prefs.stt_variant.value[0])
            print_heading(f"Select Whisper transcription model {value}")
            for i, item in enumerate(SttVariant):
                hotkey_string = make_hotkey_string(f"{i+1}")
                printt(f"{hotkey_string} {item.id} {COL_DIM}({item.description})")
            printt()
            AppUtil.show_hint_if_necessary(state.prefs, HINT_TRANSCRIPTION)

            hotkey = ask_hotkey()
            if not hotkey or not hotkey.isdigit():
                break
            index = int(hotkey) - 1
            if index < 0 or index >= len(SttVariant):
                break
            if value:
                stt_items = list(SttVariant)
                state.prefs.stt_variant = stt_items[index]
                Stt.set_variant(state.prefs.stt_variant)
                printt_set(f"Whisper transcription model set to: {state.prefs.stt_variant.id}")
                break
