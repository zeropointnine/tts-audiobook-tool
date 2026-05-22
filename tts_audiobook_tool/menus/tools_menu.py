from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.enhance import enhance_flow
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.sound.mp3_concat import SoundConcatTranscodeUtil
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.sound.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class ToolsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def speed_handler(_: State, __: MenuItem):
            hints.show_hint_if_necessary(state.prefs, HINT_SPEED_UP)
            ToolsMenu.ask_save_speed_up_audio()

        def item_maker(_: State) -> list[MenuItem]:

            enhance_item = MenuItem(
                "Enhance a pre-existing audiobook", 
                lambda _, __: enhance_flow.ask_and_make(state)
            )
            mp3s_item = MenuItem(
                "Concatenate a directory of audio files",
                lambda _, __: SoundConcatTranscodeUtil.ask_and_concat_audio_files()
            )
            transcode_item = MenuItem(
                "Transcode an app-created FLAC to M4B, preserving custom metadata",
                lambda _, __: TranscodeUtil.ask_transcode_abr_flac_to_aac(state)
            )
            speed_item = MenuItem("Speed up voice sample", speed_handler)
            
            return [enhance_item, mp3s_item, transcode_item, speed_item]
        
        MenuUtil.menu(state, "Tools", item_maker, breadcrumb="Tools")

    @staticmethod
    def ask_save_speed_up_audio() -> None:

        def cleanup():
            SoundFileUtil.stop_sound_async()

        path = ask.ask_file_path("Enter sound clip:", "Select sound clip")
        if not path:
            return
        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            ask.ask_error(result)
            return
        sound = result

        printt("Playing audio...")
        SoundFileUtil.play_sound_async(sound)

        printt()
        s ="Enter new speed as a percentage (range: 50-200) (50 = half as fast, 200 = twice as fast)"
        printt(s)
        inp = ask.ask()
        if not inp:
            cleanup()
            return
        if not inp.isdigit():
            ask.ask_error("Bad value")
        percent = float(inp)
        if not (50 <= percent <= 200):
            ask.ask_error("Out of range")
        result = SoundExtraUtil.speed_up_audio(sound, percent / 100)
        if isinstance(result, str):
            ask.ask_error(result)
            cleanup()
            return
        new_sound = result

        path_obj = Path(path)
        new_stem = f"{path_obj.stem}_speed_{int(percent)}_percent"
        new_path = str(path_obj.with_stem(new_stem).with_suffix(".flac"))

        err = SoundFileUtil.save_flac(new_sound, new_path)
        if err:
            ask.ask_error(err)
            cleanup()
            return

        printt("Playing new version...")
        SoundFileUtil.play_sound_async(new_sound)

        printt()
        printt(f"Saved new file to: {new_path}")
        ask.ask_enter_to_continue()
        cleanup()
