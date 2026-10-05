from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.higgs_v2_base_model import HiggsV2BaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelHiggsV2Menu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            items = []
             
            item = ModelMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("higgs_v2_local", "temperature"),
                default_value=HiggsV2BaseModel.DEFAULT_TEMPERATURE,
                min_value=0.01,
                max_value=2.0
            )
            items.append(item)

            items.append(
                ModelMenuShared.make_top_p_item(
                    state=state,
                    target=SettingRef("higgs_v2_local", "top_p"),
                    default_value=HiggsV2BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                ModelMenuShared.make_top_k_item(
                    state=state,
                    target=SettingRef("higgs_v2_local", "top_k"),
                    default_value=HiggsV2BaseModel.DEFAULT_TOP_K
                )
            )

            items.append(ModelMenuShared.make_seed_item(state, SettingRef("higgs_v2_local", "seed")))
            return items
        
        ModelMenuShared.menu_wrapper(state, make_items)
