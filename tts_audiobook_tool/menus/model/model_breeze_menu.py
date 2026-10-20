from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

MODEL_ID = "breeze_tts_2_local"


class ModelBreezeMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            # Sampling values apply to both the backbone and the depth decoder (as upstream)
            return [
                ModelMenuShared.make_temperature_item(
                    state=state,
                    target=SettingRef(MODEL_ID, "temperature"),
                    default_value=BreezeBaseModel.TEMPERATURE_DEFAULT,
                    min_value=BreezeBaseModel.TEMPERATURE_MIN,
                    max_value=BreezeBaseModel.TEMPERATURE_MAX,
                ),
                ModelMenuShared.make_top_p_item(
                    state=state,
                    target=SettingRef(MODEL_ID, "top_p"),
                    default_value=BreezeBaseModel.TOP_P_DEFAULT,
                ),
                ModelMenuShared.make_top_k_item(
                    state=state,
                    target=SettingRef(MODEL_ID, "top_k"),
                    default_value=BreezeBaseModel.TOP_K_DEFAULT,
                ),
                ModelMenuShared.make_repetition_penalty_item(
                    state=state,
                    target=SettingRef(MODEL_ID, "repetition_penalty"),
                    default_value=BreezeBaseModel.REPETITION_PENALTY_DEFAULT,
                ),
                MenuUtil.make_number_item(
                    state=state,
                    target=SettingRef(MODEL_ID, "cfg"),
                    base_label="CFG",
                    default_value=BreezeBaseModel.CFG_DEFAULT,
                    is_minus_one_default=True,
                    num_decimals=1,
                    prompt=f"Enter CFG scale {COL_DIM}(used only by voices with instructions)",
                    min_value=BreezeBaseModel.CFG_MIN,
                    max_value=BreezeBaseModel.CFG_MAX,
                ),
                ModelMenuShared.make_seed_item(state, SettingRef(MODEL_ID, "seed")),
            ]

        ModelMenuShared.menu_wrapper(state, make_items)
