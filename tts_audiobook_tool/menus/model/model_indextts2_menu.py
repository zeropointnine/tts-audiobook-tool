from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelIndexTts2Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        project = state.project

        def make_fp16_item(_) -> str:
            value = make_parameter_value_string(project.get_model_setting('indextts2_local', 'use_fp16'), IndexTts2BaseModel.DEFAULT_USE_FP16)
            return f"FP16 (smaller memory footprint) {make_currently_string(value)}"

        # Menu
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(make_fp16_item, lambda _, __: ModelIndexTts2Menu.fp16_menu(state))
            )

            item = ModelMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("indextts2_local", "temperature"),
                default_value=IndexTts2BaseModel.DEFAULT_TEMPERATURE,
                min_value=0.01,
                max_value=2.0
            )
            items.append(item)

            items.append(
                ModelMenuShared.make_top_p_item(
                    state=state,
                    target=SettingRef("indextts2_local", "top_p"),
                    default_value=IndexTts2BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                ModelMenuShared.make_top_k_item(
                    state=state,
                    target=SettingRef("indextts2_local", "top_k"),
                    default_value=IndexTts2BaseModel.DEFAULT_TOP_K
                )
            )

            items.append(ModelMenuShared.make_seed_item(state, SettingRef("indextts2_local", "seed")))

            return items

        ModelMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def fp16_menu(state: State) -> None:

        def on_item(_: State, item: MenuItem) -> bool:
            if state.project.get_model_setting('indextts2_local', 'use_fp16') != item.data:
                state.project.set_model_setting('indextts2_local', 'use_fp16', item.data)
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project)
            print_feedback(f"FP16 set to:", str(state.project.get_model_setting('indextts2_local', 'use_fp16')))
            return True

        items = [
            MenuItem("True", on_item, data=True),
            MenuItem("False", on_item, data=False)
        ]
        MenuUtil.menu(
            state,
            "FP16",
            items,
            one_shot=True
        )
