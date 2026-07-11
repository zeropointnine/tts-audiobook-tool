from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.higgs_v2_base_model import HiggsV2BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceHiggsV2Menu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            items = [
                VoiceMenuShared.make_manage_voice_samples_item(state, TtsModelType.HIGGS_V2)
            ]
             
            item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="higgs_temperature",
                default_value=HiggsV2BaseModel.DEFAULT_TEMPERATURE,
                min_value=0.01,
                max_value=2.0
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="higgs_top_p",
                    default_value=HiggsV2BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="higgs_top_k",
                    default_value=HiggsV2BaseModel.DEFAULT_TOP_K
                )
            )

            items.append(VoiceMenuShared.make_seed_item(state, "higgs_seed"))
            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)
