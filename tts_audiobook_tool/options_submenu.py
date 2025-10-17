from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.mp3_concat import Mp3ConcatTranscodeUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.real_time_submenu import RealTimeSubmenu
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
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

            print_heading("Options/Tools:")
            printt(f"{make_hotkey_string('1')} Real-time playback")
            printt(f"{make_hotkey_string('2')} Enhance existing audiobook file {COL_DIM}(experimental)")
            printt(f"{make_hotkey_string('3')} Transcode and concatenate a directory of MP3 files to AAC/M4A")
            printt(f"{make_hotkey_string('4')} Transcode an app-created FLAC to AAC/M4A, preserving custom metadata")
            printt(f"{make_hotkey_string('5')} Speed up voice sample")
            printt(f"{make_hotkey_string('6')} Whisper transcription model {make_currently_string(state.project.stt_variant.value)}")
            printt(f"{make_hotkey_string('7')} Attempt to unload models {vram_label}")
            printt(f"{make_hotkey_string('8')} Reset contextual hints")
            printt()

            hotkey = ask_hotkey()
            match hotkey:
                case "1":
                    RealTimeSubmenu.submenu(state)
                case "2":
                    SttFlow.ask_and_make(state.prefs)
                case "3":
                    Mp3ConcatTranscodeUtil.ask_mp3_dir()
                case "4":
                    TranscodeUtil.ask_transcode_abr_flac_to_aac(state)
                case "5":
                    AppUtil.show_hint_if_necessary(state.prefs, HINT_SPEED_UP)
                    OptionsSubmenu.ask_save_speed_up_audio()
                case "6":
                    OptionsSubmenu.ask_transcription_model(state)
                case "7":
                    Tts.clear_all_models()
                    printt_set("Finished")
                case "8":
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

    @staticmethod
    def ask_save_speed_up_audio() -> None:

        def cleanup():
            SoundFileUtil.stop_sound_async()

        path = ask_file_path("Enter sound clip:", "Select sound clip")
        if not path:
            return
        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            ask_error(result)
            return
        sound = result

        printt("Playing audio...")
        SoundFileUtil.play_sound_async(sound)

        printt()
        s ="Enter new speed as a percentage (range: 50-200) (50 = half as fast, 200 = twice as fast)"
        printt(s)
        inp = ask()
        if not inp:
            cleanup()
            return
        if not inp.isdigit():
            ask_error("Bad value")
        percent = float(inp)
        if not (50 <= percent <= 200):
            ask_error("Out of range")
        result = SoundUtil.speed_up_audio(sound, percent / 100)
        if isinstance(result, str):
            ask_error(result)
            cleanup()
            return
        new_sound = result

        path_obj = Path(path)
        new_stem = f"{path_obj.stem}_speed_{int(percent)}_percent"
        new_path = str(path_obj.with_stem(new_stem).with_suffix(".flac"))

        err = SoundFileUtil.save_flac(new_sound, new_path)
        if err:
            ask_error(err)
            cleanup()
            return

        printt("Playing new version...")
        SoundFileUtil.play_sound_async(new_sound)

        printt()
        printt(f"Saved new file to: {new_path}")
        ask_continue()
        cleanup()


