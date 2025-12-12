from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.mp3_concat import Mp3ConcatTranscodeUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt_flow import SttFlow
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class ToolsMenu:

    @staticmethod
    def menu(state: State) -> None:

        enhance_item = MenuItem(
            f"Enhance existing audiobook file {COL_DIM}(experimental)",
            lambda _, __: SttFlow.ask_and_make(state.prefs)
        )

        mp3s_item = MenuItem(
            "Transcode and concatenate a directory of MP3 files to AAC/M4A",
            lambda _, __: Mp3ConcatTranscodeUtil.ask_mp3_dir()
        )

        transcode_item = MenuItem(
            "Transcode an app-created FLAC to AAC/M4A, preserving custom metadata",
            lambda _, __: TranscodeUtil.ask_transcode_abr_flac_to_aac(state)
        )

        def speed_handler(_: State, __: MenuItem):
            Hint.show_hint_if_necessary(state.prefs, HINT_SPEED_UP)
            ToolsMenu.ask_save_speed_up_audio()

        speed_item = MenuItem("Speed up voice sample", speed_handler)

        items = [enhance_item, mp3s_item, transcode_item, speed_item]
        MenuUtil.menu(state, "Tools:", items)

    @staticmethod
    def ask_save_speed_up_audio() -> None:

        def cleanup():
            SoundFileUtil.stop_sound_async()

        path = AskUtil.ask_file_path("Enter sound clip:", "Select sound clip")
        if not path:
            return
        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            AskUtil.ask_error(result)
            return
        sound = result

        printt("Playing audio...")
        SoundFileUtil.play_sound_async(sound)

        printt()
        s ="Enter new speed as a percentage (range: 50-200) (50 = half as fast, 200 = twice as fast)"
        printt(s)
        inp = AskUtil.ask()
        if not inp:
            cleanup()
            return
        if not inp.isdigit():
            AskUtil.ask_error("Bad value")
        percent = float(inp)
        if not (50 <= percent <= 200):
            AskUtil.ask_error("Out of range")
        result = SoundUtil.speed_up_audio(sound, percent / 100)
        if isinstance(result, str):
            AskUtil.ask_error(result)
            cleanup()
            return
        new_sound = result

        path_obj = Path(path)
        new_stem = f"{path_obj.stem}_speed_{int(percent)}_percent"
        new_path = str(path_obj.with_stem(new_stem).with_suffix(".flac"))

        err = SoundFileUtil.save_flac(new_sound, new_path)
        if err:
            AskUtil.ask_error(err)
            cleanup()
            return

        printt("Playing new version...")
        SoundFileUtil.play_sound_async(new_sound)

        printt()
        printt(f"Saved new file to: {new_path}")
        AskUtil.ask_enter_to_continue()
        cleanup()
