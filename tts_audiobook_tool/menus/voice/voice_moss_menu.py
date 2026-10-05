from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceMossMenu:

    @staticmethod
    def menu(state: State) -> None:
        def make_items(_: State) -> list[MenuItem]:
            items: list[MenuItem] = []
            VoiceMossShared.append_voice_items(items, state, TtsModelType.require_by_id("moss_local"))
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
