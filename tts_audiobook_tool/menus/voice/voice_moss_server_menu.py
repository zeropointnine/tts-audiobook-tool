from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice import VoiceMenuShared


class VoiceMossServerMenu:
    """
    Shared server-based MOSS settings menu for Delay and Local variants.
    """
    
    @staticmethod
    def menu(state: State, model_type: TtsModelType) -> None:
        configs = {
            TtsModelType.MOSS_DELAY_SERVER: MossConfigs.DELAY,
            TtsModelType.MOSS_LOCAL_SERVER: MossConfigs.LOCAL,
        }
        try:
            config = configs[model_type]
        except KeyError as e:
            raise ValueError(f"Unsupported MOSS server type: {model_type}") from e

        def make_items(_: State) -> list[MenuItem]:

            items = []
            # Rem, we CAN utilize local voice clone path because it gets transmitted
            # as data uri (many other sgl-omni per-model apis do NOT support this)
            VoiceMossShared.append_voice_items(items, state, model_type)
            items.append(VoiceMossShared.make_temperature_item(state, config))
            items.append(VoiceMossShared.make_audio_top_p_item(state, config))
            items.append(VoiceMossShared.make_audio_top_k_item(state, config))

            item = VoiceMenuShared.make_seed_item(state, "moss_seed", add_batch_warning=True)
            items.append(item)

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
