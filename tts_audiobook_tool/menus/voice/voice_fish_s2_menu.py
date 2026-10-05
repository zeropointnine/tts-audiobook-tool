from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceFishS2Menu:

    @staticmethod
    def menu(state: State) -> None:
        def make_items(_: State) -> list[MenuItem]:
            return VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id("fish_s2_local"))

        VoiceMenuShared.menu_wrapper(state, make_items)
