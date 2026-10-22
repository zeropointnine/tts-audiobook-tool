from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelChatterboxMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        
        def make_items(_: State) -> list[MenuItem]:
            model_type: ChatterboxType = state.project.get_model_setting('chatterbox_local', 'type')

            def item(name: str, label: str) -> MenuItem:
                return ModelCatalogMenu.make_local_parameter_item(state, "chatterbox_local", name, label)

            items = [MenuItem(make_model_type_label, lambda _, __: ask_type(state))]
            items.append(item("temperature", "Temperature"))
            if model_type.is_multilingual:
                items.append(item("exaggeration", "Exaggeration"))
                items.append(item("cfg", "CFG/pace"))
            items.append(item("top_p", "Top-P"))
            if model_type == ChatterboxType.TURBO:
                items.append(item("turbo_top_k", "Top-K"))
            # Each variant has its own repetition penalty
            qualifier = model_type.label.removeprefix("Chatterbox-") if model_type.is_multilingual else "Turbo"
            items.append(item(
                ChatterboxBaseModel.repetition_penalty_setting(model_type), f"Repetition penalty ({qualifier})"))
            items.append(ModelMenuShared.make_seed_item(state, SettingRef("chatterbox_local", "seed")))
            return items

        ModelMenuShared.menu_wrapper(state, make_items)

def make_model_type_label(state: State) -> str:
    return make_menu_label("Select model", state.project.get_model_setting('chatterbox_local', 'type').label)

def ask_type(state: State) -> None:

    def on_select(value: ChatterboxType) -> None:
        state.project.set_model_setting('chatterbox_local', 'type', value)
        state.project.save()
        # Sync static value
        Tts.set_model_params_using_project(state.project) 
        print_feedback("Model set to:", state.project.get_model_setting('chatterbox_local', 'type').label)

    MenuUtil.options_menu(
        state=state,
        heading_text="Chatterbox model",
        labels=[item.label for item in list(ChatterboxType)],
        values=[item for item in list(ChatterboxType)],
        current_value=state.project.get_model_setting('chatterbox_local', 'type'),
        default_value=list(ChatterboxType)[0],
        on_select=on_select
    )
