from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class VoiceMossShared:

    @staticmethod
    def append_voice_items(
            items: list[MenuItem], state: State, model_type: TtsModelType
    ) -> None:
        items.extend(
            VoiceMenuShared.make_voice_sample_items(state, model_type)
        )
