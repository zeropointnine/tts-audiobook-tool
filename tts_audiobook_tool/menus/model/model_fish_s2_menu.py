from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelFishS2Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []

            items.append(
                MenuItem(
                    make_menu_label("Torch compile", state.project.get_model_setting('fish_s2_local', 'compile_enabled')),
                    lambda _, __: ModelFishS2Menu.compile_menu(state),
                )
            )

            item = MenuItem(
                ModelMenuShared.make_rolling_continuation_label(state.project.get_model_setting('fish_s2_local', 'rolling_cont')),
                lambda _, __: ModelMenuShared.ask_rolling_continuation(
                    state=state,
                    target=SettingRef("fish_s2_local", "rolling_cont"),
                    max_value=FishS2BaseModel.ROLLING_CONTINUATION_MAX_LENGTH,
                    qualifier_line="Fish S2 rolling continuation requires batch size 1."
                )
            )
            items.append(item)

            temperature_item = ModelMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("fish_s2_local", "temperature"),
                default_value=FishS2BaseModel.TEMPERATURE_DEFAULT,
                min_value=FishS2BaseModel.TEMPERATURE_MIN,
                max_value=FishS2BaseModel.TEMPERATURE_MAX
            )
            items.append(temperature_item)

            items.append(
                ModelMenuShared.make_top_p_item(
                    state=state,
                    target=SettingRef("fish_s2_local", "top_p"),
                    default_value=FishS2BaseModel.TOP_P_DEFAULT
                )
            )

            items.append(
                ModelMenuShared.make_top_k_item(
                    state=state,
                    target=SettingRef("fish_s2_local", "top_k"),
                    default_value=FishS2BaseModel.TOP_K_DEFAULT
                )
            )

            items.append(
                ModelMenuShared.make_seed_item(state, SettingRef("fish_s2_local", "seed"))
            )

            return items
        
        ModelMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def compile_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.project.get_model_setting('fish_s2_local', 'compile_enabled') != value:
                state.project.set_model_setting('fish_s2_local', 'compile_enabled', value)
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project)
            print_feedback(f"Set to:", str(state.project.get_model_setting('fish_s2_local', 'compile_enabled')))

        MenuUtil.options_menu(
            state=state,
            heading_text="Torch compile",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.get_model_setting('fish_s2_local', 'compile_enabled'),
            default_value=True,
            on_select=on_select
        )
