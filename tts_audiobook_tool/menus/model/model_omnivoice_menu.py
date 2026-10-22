from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelOmniVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_target_label(_) -> str:
            target = state.project.get_model_setting('omnivoice_local', 'target') or OmniVoiceBaseModel.DEFAULT_REPO_ID
            is_default = (target == OmniVoiceBaseModel.DEFAULT_REPO_ID)
            if is_default:
                label = f"{COL_DIM}(optional)"
            else:
                value = ellipsize_path_for_menu(target)
                label = make_currently_string(value)
            return f"Custom model {label}"

        def on_clear_model_target(s: State, __: MenuItem) -> None:
            s.project.set_model_setting('omnivoice_local', 'target', "")
            s.project.save()
            Tts.set_model_params_using_project(s.project)
            _ = ModelWorker.clear_models_if_running_blocking()
            print_feedback("Cleared, will use default model")

        def make_items(_: State) -> list[MenuItem]:
            items = []

            items.append(
                MenuItem(make_target_label, lambda _, __: ask_target(state))
            )
            if state.project.get_model_setting('omnivoice_local', 'target'):
                items.append(MenuItem("Clear custom model", on_clear_model_target))

            for name, label in (("num_step", "Steps"), ("speed", "Speed"), ("cfg", "CFG")):
                items.append(ModelCatalogMenu.make_local_parameter_item(state, "omnivoice_local", name, label))

            items.append(
                ModelMenuShared.make_seed_item(state, SettingRef("omnivoice_local", "seed"))
            )

            return items

        ModelMenuShared.menu_wrapper(state, make_items)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_target(state: State) -> None:
    project = state.project
    model_name = project.get_tts_model_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"k2-fsa/OmniVoice\" or \"/path/to/local/checkpoint\""
    ModelMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.get_model_setting('omnivoice_local', 'target'),
        callback=lambda _, target: apply_target(state, target)
    )

def apply_target(state: State, target: str) -> None:
    project = state.project
    previous_target = project.get_model_setting('omnivoice_local', 'target')

    def revert() -> None:
        project.set_model_setting('omnivoice_local', 'target', previous_target)
        _ = ModelWorker.clear_models_if_running_blocking()

    project.set_model_setting('omnivoice_local', 'target', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"Failed to load OmniVoice model: {error}")
        return

    project.save()
    print_feedback("Model set:", target)
