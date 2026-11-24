from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared, TargetPromptSpec

class ModelOmniVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_target_label(_) -> str:
            target = state.project.get_model_target('omnivoice_local') or OmniVoiceBaseModel.DEFAULT_TARGET
            is_default = (target == OmniVoiceBaseModel.DEFAULT_TARGET)
            if is_default:
                label = f"{COL_DIM}(optional)"
            else:
                value = ellipsize_path_for_menu(target.value)
                label = make_currently_string(value)
            return f"Custom model {label}"

        def make_items(_: State) -> list[MenuItem]:
            items = []

            items.append(
                MenuItem(make_target_label, lambda _, __: ask_target(state))
            )

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
    ModelMenuShared.ask_target(
        project=project,
        spec=TargetPromptSpec(subject=f"{model_name} model", hf_example=OmniVoiceBaseModel.DEFAULT_REPO_ID),
        current_target=project.get_model_target('omnivoice_local'),
        callback=lambda _, target: apply_target(state, target),
        on_clear=lambda: on_clear_model_target(state),
    )


def on_clear_model_target(state: State) -> None:
    state.project.set_model_target('omnivoice_local', None)
    state.project.save()
    Tts.set_model_params_using_project(state.project)
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared, will use default model")

def apply_target(state: State, target: ModelTarget) -> None:
    project = state.project
    previous_target = project.get_model_target('omnivoice_local')

    def revert() -> None:
        project.set_model_target('omnivoice_local', previous_target)
        _ = ModelWorker.clear_models_if_running_blocking()

    project.set_model_target('omnivoice_local', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"Failed to load OmniVoice model: {error}")
        return

    project.save()
    print_feedback("Model set:", target.value)
