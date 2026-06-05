from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.moss_server_base_model import MossServerBaseModel
from tts_audiobook_tool.menus.voice import VoiceMenuShared


class VoiceMossServerMenu:
    """
    Server-based version of MOSS setting menu
    Uses the same project settings as the local version ("delay" variant).
    """
    
    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:

            items = []
            VoiceMossShared.append_voice_items(items, state)
            items.append(VoiceMossShared.make_temperature_item(state, MossServerBaseModel.CONFIG))
            items.append(VoiceMossShared.make_audio_top_p_item(state, MossServerBaseModel.CONFIG))
            items.append(VoiceMossShared.make_audio_top_k_item(state, MossServerBaseModel.CONFIG))

            item = VoiceMenuShared.make_seed_item(state, "moss_seed", add_batch_warning=True)
            items.append(item)

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
