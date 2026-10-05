from collections.abc import Callable
from typing import Any

from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import TtsInspected
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

class ModelQwen3Menu:
    """
    Note, menu requires knowing qwen3 model type,
    which requires model being instantiated
    (unlike the other static model menus, which do not require this)
    """

    @staticmethod
    def menu(state: State, inspection: TtsInspected) -> None:
        metadata = inspection.metadata or {}

        def get_model_type() -> str:
            return str(metadata.get("model_type", state.project.get_model_setting('qwen3tts_local', 'model_type')))

        def get_generate_defaults() -> dict[str, Any]:
            value = metadata.get("generate_defaults", {})
            return value if isinstance(value, dict) else {}

        def apply_target_and_refresh(target: str) -> None:
            def refresh(updated_inspection: TtsInspected) -> None:
                nonlocal metadata
                metadata = updated_inspection.metadata or {}

            apply_model_and_validate(state, target, on_applied=refresh)

        def clear_target_and_refresh(_: State, item: MenuItem) -> None:
            nonlocal metadata
            on_clear_model_target(state, item)
            # The built-in default is a Base model. Discard metadata from the
            # previously inspected custom target before this menu rerenders.
            metadata = {"model_type": "base"}

        def make_target_label(_) -> str:
            model_type = get_model_type()
            extra_suffix = f" {COL_DIM}(model type: {COL_ACCENT}{model_type}{COL_DIM})"
            return ModelMenuShared.make_target_label(
                label_prefix="Select Qwen3-TTS model",
                target=state.project.get_model_setting('qwen3tts_local', 'target'),
                default_target=Qwen3BaseModel.DEFAULT_REPO_ID,
                remove_prefixes=["Qwen/"],
                extra_suffix=extra_suffix,
            )

        def make_items(_: State) -> list[MenuItem]:
            generate_defaults = get_generate_defaults()
            items = []

            # Model, clear model
            items.append(
                MenuItem(
                    make_target_label,
                    lambda _, __: model_target_submenu(state, apply_target_and_refresh),
                )
            )
            if state.project.get_model_setting('qwen3tts_local', 'target'):
                items.append(
                    MenuItem("Clear custom model", clear_target_and_refresh)
                )

            # Always show rolling cont setting even though requires type 'base' and batch 1
            item = MenuItem(
                ModelMenuShared.make_rolling_continuation_label(state.project.get_model_setting('qwen3tts_local', 'rolling_cont')),
                lambda _, __: ModelMenuShared.ask_rolling_continuation(
                    state=state,
                    target=SettingRef("qwen3tts_local", "rolling_cont"),
                    max_value=Qwen3BaseModel.ROLLING_CONTINUATION_MAX_LENGTH,
                    qualifier_line="Qwen3-TTS model must be of type \"base\", and batch size must be 1."
                )
            )
            items.append(item)

            default_temp = generate_defaults.get(
                "temperature", Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT
            )
            item = ModelMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("qwen3tts_local", "temperature"),
                default_value=default_temp,
                min_value=Qwen3BaseModel.TEMPERATURE_MIN,
                max_value=Qwen3BaseModel.TEMPERATURE_MAX
            )
            items.append(item)

            default_top_p = generate_defaults.get(
                "top_p", Qwen3BaseModel.TOP_P_DEFAULT
            )
            item = ModelMenuShared.make_top_p_item(
                state=state,
                target=SettingRef("qwen3tts_local", "top_p"),
                default_value=default_top_p
            )
            items.append(item)

            default_top_k = generate_defaults.get(
                "top_k", Qwen3BaseModel.TOP_K_DEFAULT
            )
            item = ModelMenuShared.make_top_k_item(
                state=state,
                target=SettingRef("qwen3tts_local", "top_k"),
                default_value=default_top_k
            )
            items.append(item)

            default_rp = generate_defaults.get(
                "repetition_penalty", Qwen3BaseModel.REPETITION_PENALTY_DEFAULT
            )
            item = ModelMenuShared.make_repetition_penalty_item(
                state=state,
                target=SettingRef("qwen3tts_local", "repetition_penalty"),
                default_value=default_rp
            )
            items.append(item)

            items.append(ModelMenuShared.make_seed_item(state, SettingRef("qwen3tts_local", "seed"), add_batch_warning=True))
            return items

        ModelMenuShared.menu_wrapper(state, make_items)

# ---

def model_target_submenu(
    state: State,
    apply_target: Callable[[str], None] | None = None,
) -> None:
    resolved_apply_target = apply_target
    if resolved_apply_target is None:
        resolved_apply_target = lambda target: apply_model_and_validate(state, target)

    ModelMenuShared.target_submenu(
        state=state,
        heading="Select Qwen3-TTS model",
        preset_targets=Qwen3BaseModel.PRESET_REPO_IDS,
        current_target=state.project.get_model_setting('qwen3tts_local', 'target'),
        default_target=Qwen3BaseModel.DEFAULT_REPO_ID,
        ask_custom_target=lambda: ask_target(state, resolved_apply_target),
        apply_target=resolved_apply_target,
    )

def ask_target(
    state: State,
    apply_target: Callable[[str], None] | None = None,
) -> None:
    project = state.project
    resolved_apply_target = apply_target
    if resolved_apply_target is None:
        resolved_apply_target = lambda target: apply_model_and_validate(state, target)

    model_name = project.get_tts_model_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"zeropointnine/Darwin-TTS-1.7B-Cross-Qwen3Tokenizer\" or \"/path/to/checkpoint\""

    ModelMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.get_model_setting('qwen3tts_local', 'target'),
        callback=lambda _, target: resolved_apply_target(target)
    )

def apply_model_and_validate(
    state: State,
    target: str,
    on_applied: Callable[[TtsInspected], None] | None = None,
) -> None:
    project = state.project

    previous_target = project.get_model_setting('qwen3tts_local', 'target')
    previous_model_type = project.get_model_setting('qwen3tts_local', 'model_type')
    previous_speaker_id = project.get_model_setting('qwen3tts_local', 'speaker_id')

    def revert() -> None:
        project.set_model_setting('qwen3tts_local', 'target', previous_target)
        project.set_model_setting('qwen3tts_local', 'model_type', previous_model_type)
        project.set_model_setting('qwen3tts_local', 'speaker_id', previous_speaker_id)
        _ = ModelWorker.clear_models_if_running_blocking()

    project.set_model_setting('qwen3tts_local', 'target', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"\nContents at {target} appear to be invalid:\n{error}")
        return

    metadata = inspection.metadata or {}
    inspected_type = str(metadata.get("model_type", ""))
    if not bool(metadata.get("is_model_type_supported", True)):
        ask.ask_error(f"Unsupported type: {inspected_type}")
        revert()
        ask.ask_enter_to_continue()
        return
    if project.get_model_setting('qwen3tts_local', 'speaker_id'):
        project.set_model_setting('qwen3tts_local', 'speaker_id', "")
    project.set_model_setting('qwen3tts_local', 'model_type', inspected_type)
    project.save()
    if on_applied is not None:
        on_applied(inspection)
    print_feedback("Model set:", target)
    ask.ask_enter_to_continue()

def on_clear_model_target(state: State, __: MenuItem) -> None:
    state.project.set_model_setting('qwen3tts_local', 'target', "")
    state.project.set_model_setting('qwen3tts_local', 'model_type', "")
    state.project.save()
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared, will use default model")
