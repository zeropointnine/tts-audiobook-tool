from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.mira_base_model import MiraBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelMiraMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        def make_items(_: State) -> list[MenuItem]:
            items = []

            item = ModelMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("mira_local", "temperature"),
                default_value=MiraBaseModel.TEMPERATURE_DEFAULT,
                min_value=MiraBaseModel.TEMPERATURE_MIN,
                max_value=MiraBaseModel.TEMPERATURE_MAX
            )
            items.append(item)

            item = ModelMenuShared.make_top_p_item(
                state=state,
                target=SettingRef("mira_local", "top_p"),
                default_value=MiraBaseModel.TOP_P_DEFAULT
            )
            items.append(item)

            item = ModelMenuShared.make_top_k_item(
                state=state,
                target=SettingRef("mira_local", "top_k"),
                default_value=MiraBaseModel.TOP_K_DEFAULT
            )
            items.append(item)

            item = ModelMenuShared.make_repetition_penalty_item(
                state=state,
                target=SettingRef("mira_local", "repetition_penalty"),
                default_value=MiraBaseModel.REPETITION_PENALTY_DEFAULT
            )
            items.append(item)

            items.append(ModelMenuShared.make_seed_item(state, SettingRef("mira_local", "seed"), add_batch_warning=True))
            return items
        
        ModelMenuShared.menu_wrapper(state, make_items)
