from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelVibeVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        project = state.project

        def make_model_target_label(_) -> str:
            return ModelMenuShared.make_target_label(
                label_prefix="Select model",
                target=project.get_model_setting('vibevoice_local', 'target'),
                default_target=VibeVoiceBaseModel.DEFAULT_REPO_ID,
            )

        def make_items(_: State) -> list[MenuItem]:

            items = []

            # Model
            items.append(
                MenuItem(
                    make_model_target_label,
                    lambda _, __: target_submenu(state),
                )
            )
            if state.project.get_model_setting('vibevoice_local', 'target'):
                items.append(
                    MenuItem("Clear custom model", lambda _, __: clear_custom_model(state))
                )

            # Other config
            item = MenuUtil.make_number_item(
                state=state,
                target=SettingRef("vibevoice_local", "cfg"),
                base_label="CFG",
                default_value=VibeVoiceBaseModel.CFG_DEFAULT,
                is_minus_one_default=True,
                num_decimals=2,
                prompt=f"Enter CFG {COL_DIM}({VibeVoiceBaseModel.CFG_MIN} to {VibeVoiceBaseModel.CFG_MAX}):",
                min_value=VibeVoiceBaseModel.CFG_MIN,
                max_value=VibeVoiceBaseModel.CFG_MAX
            )
            items.append(item)

            items.append(
                MenuUtil.make_number_item(
                    state=state,
                    target=SettingRef("vibevoice_local", "steps"),
                    base_label="Steps",
                    default_value=VibeVoiceBaseModel.DEFAULT_NUM_STEPS,
                    is_minus_one_default=True,
                    num_decimals=0,
                    prompt=f"Enter num steps {COL_DIM}(1-30){COL_DEFAULT}:",
                    min_value=1,
                    max_value=30
                )
            )

            items.append(
                ModelMenuShared.make_seed_item(state, SettingRef("vibevoice_local", "seed"))
            )
            return items

        ModelMenuShared.menu_wrapper(state, make_items)

# ---

def target_submenu(state: State) -> None:
    ModelMenuShared.target_submenu(
        state=state,
        heading="Select VibeVoice model",
        preset_targets=VibeVoiceBaseModel.PRESET_REPO_IDS,
        current_target=state.project.get_model_setting('vibevoice_local', 'target'),
        default_target=VibeVoiceBaseModel.DEFAULT_REPO_ID,
        ask_custom_target=lambda: ask_model_target(state),
        apply_target=lambda target: apply_model_and_validate(state, target),
    )

def ask_model_target(state: State) -> None:
    project = state.project

    model_name = project.get_tts_model_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"vibevoice/VibeVoice-7B\"; \"/path/to/checkpoint\""
    if project.get_model_setting('vibevoice_local', 'target'):
        prompt += f"\n{COL_DIM}(currently: {project.get_model_setting('vibevoice_local', 'target')})"

    ModelMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.get_model_setting('vibevoice_local', 'target'),
        callback=lambda _, target: apply_model_and_validate(state, target)
    )

def apply_model_and_validate(state: State, target: str) -> None:
    project = state.project

    previous_target = project.get_model_setting('vibevoice_local', 'target')
    project.set_model_setting('vibevoice_local', 'target', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    # Preset targets are known-good repos with a single fixed architecture,
    # so there is nothing to learn from loading the model here. Custom
    # targets are arbitrary repos/paths, so validate those by loading.
    if target in VibeVoiceBaseModel.PRESET_REPO_IDS:
        project.save()
        print_feedback("\nCustom model set:", target)
        return

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        project.set_model_setting('vibevoice_local', 'target', previous_target)
        _ = ModelWorker.clear_models_if_running_blocking()
        ask.ask_error(f"\n{error}")
        return

    project.save()
    print_feedback("\nCustom model set:", target)

def clear_custom_model(state: State) -> None:
    project = state.project
    project.set_model_setting('vibevoice_local', 'target', "")
    project.save()
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared, will use default model")
