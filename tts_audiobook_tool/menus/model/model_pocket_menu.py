from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelPocketMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_language_label(_) -> str:
            lang = state.project.get_model_setting('pocket_local', 'model_code')
            if lang:
                label = make_currently_string(lang)
            else:
                label = make_currently_string(PocketBaseModel.DEFAULT_LANGUAGE)
            return f"Pocket model {label}"

        def make_items(_: State) -> list[MenuItem]:

            items = []

            items.append(
                MenuItem(
                    make_language_label, 
                    lambda _, __: ask_language(state),
                )
            )

            item = ModelMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("pocket_local", "temperature"),
                default_value=PocketBaseModel.DEFAULT_TEMPERATURE,
                min_value=PocketBaseModel.TEMPERATURE_MIN,
                max_value=PocketBaseModel.TEMPERATURE_MAX
            )
            items.append(item)

            items.append(
                ModelMenuShared.make_seed_item(state, SettingRef("pocket_local", "seed"))
            )

            return items

        ModelMenuShared.menu_wrapper(state, make_items)

# ---

def ask_language(state: State) -> None:

    languages = PocketBaseModel.LANGUAGES
    default = PocketBaseModel.DEFAULT_LANGUAGE

    def on_select(lang: str) -> None:

        previous = state.project.get_model_setting('pocket_local', 'model_code')
        if lang == previous:
            return

        # Validation step arguably non-essential since we're using a controlled, hard-list, but yea
        printt(f"{COL_DIM_ITALICS}Validating... ")
        printt()
        state.project.set_model_setting('pocket_local', 'model_code', lang)
        state.project.save()
        _ = ModelWorker.clear_models_if_running_blocking()
        inspection, error = ModelWorker.inspect_tts_blocking(state)
        if error or inspection is None:
            state.project.set_model_setting('pocket_local', 'model_code', previous)
            state.project.save()
            _ = ModelWorker.clear_models_if_running_blocking()
            ask.ask_error(f"\n{error}")
        printt(f"{COL_DIM_ITALICS}OK ")
        printt()

    MenuUtil.options_menu(
        state=state,
        heading_text="Select Pocket TTS model",
        labels=languages,
        values=languages,
        subheading="Ensure your voice clone audio language matches the model you select below",
        current_value=state.project.get_model_setting('pocket_local', 'model_code') or default,
        default_value=default,
        on_select=on_select,
    )

