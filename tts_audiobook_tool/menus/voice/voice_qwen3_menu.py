from collections.abc import Callable
from typing import Any

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import TtsInspected
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceQwen3Menu:
    """
    Note, menu requires knowing qwen3 model type,
    which requires model being instantiated
    (unlike the other model voice menus, which do not require this)
    """

    @staticmethod
    def menu(state: State, inspection: TtsInspected) -> None:
        metadata = inspection.metadata or {}

        def get_model_type() -> str:
            return str(metadata.get("model_type", state.project.qwen3_model_type))

        def get_speakers() -> list[str]:
            value = metadata.get("supported_speakers", [])
            return (
                [str(speaker) for speaker in value]
                if isinstance(value, (list, tuple))
                else []
            )

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

        def make_voice_label(_) -> str:
            if not state.project.qwen3_voice_file_name:
                currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
            else:
                value = ProjectVoiceUtil.get_voice_label(state.project)
                value = ellipsize_path_for_menu(value)
                currently = make_currently_string(value)
            return f"Select voice clone sample {currently}"

        def make_target_label(_) -> str:
            model_type = get_model_type()
            extra_suffix = f" {COL_DIM}(model type: {COL_ACCENT}{model_type}{COL_DIM})"
            return VoiceMenuShared.make_target_label(
                label_prefix="Select Qwen3-TTS model",
                target=state.project.qwen3_target,
                default_target=Qwen3BaseModel.DEFAULT_REPO_ID,
                remove_prefixes=["Qwen/"],
                extra_suffix=extra_suffix,
            )

        def make_speaker_label(_) -> str:
            speakers = get_speakers()
            has_only_one = (len(speakers) == 1)
            if has_only_one:
                speaker_id = speakers[0]
            else:
                speaker_id = state.project.qwen3_speaker_id
            value = speaker_id or "None"
            suffix = make_currently_string(value)
            if speaker_id not in speakers:
                if not speaker_id:
                    suffix = f"({COL_ERROR}required{COL_DIM})"
                else:
                    suffix += f" ({COL_ERROR}required - current id is invalid{COL_DIM})"
            return "Set speaker " + suffix

        def make_instructions_cv_label(_) -> str:
            if not state.project.qwen3_instructions:
                suffix = f"{COL_DIM}(optional)"
            else:
                value = truncate_pretty(state.project.qwen3_instructions, 40, content_color=COL_ACCENT)
                suffix = make_currently_string(value)
            return f"Instructions {suffix}"

        def make_instructions_vd_label(_) -> str:
            if not state.project.qwen3_instructions:
                suffix = make_currently_string("none", color_code=COL_ERROR)
            else:
                value = truncate_pretty(state.project.qwen3_instructions, 40, content_color=COL_ACCENT)
                suffix = make_currently_string(value)
            return f"Instructions {suffix}"

        def on_clear_instructions(_: State, __: MenuItem) -> None:
            state.project.qwen3_instructions = ""
            state.project.save()
            print_feedback("Instructions cleared")

        def on_clear_speaker(_: State, __: MenuItem) -> None:
            state.project.qwen3_speaker_id = ""
            state.project.save()
            print_feedback("Speaker cleared")

        def make_items(_: State) -> list[MenuItem]:
            model_type = get_model_type()
            speakers = get_speakers()
            generate_defaults = get_generate_defaults()
            items = []

            match model_type:
                case "base":
                    # Voice clone, clear voice clone
                    items.extend(
                        VoiceMenuShared.make_voice_sample_items(
                            state,
                            TtsModelType.QWEN3TTS,
                            no_samples_label=make_voice_label,
                        )
                    )
                case "custom_voice":
                    # Speaker id, instructions
                    items.append(
                        MenuItem(
                            make_speaker_label,
                            lambda _, __: ask_speaker_id(
                                state.project,
                                speakers,
                            ),
                        )
                    )
                    if state.project.qwen3_speaker_id:
                        items.append(
                            MenuItem("Clear speaker", on_clear_speaker)
                        )
                    items.append(
                        MenuItem(make_instructions_cv_label, lambda _, __: ask_instructions(state.project))
                    )
                    if state.project.qwen3_instructions:
                        items.append(
                            MenuItem("Clear instructions", on_clear_instructions)
                        )
                case "voice_design":
                    # Instructions
                    items.append(
                        MenuItem(make_instructions_vd_label, lambda _, __: ask_instructions(state.project))
                    )
                    if state.project.qwen3_instructions:
                        items.append(
                            MenuItem("Clear instructions", on_clear_instructions)
                        )

            # Model, clear model
            items.append(
                MenuItem(
                    make_target_label,
                    lambda _, __: model_target_submenu(state, apply_target_and_refresh),
                    superlabel=VOICE_ADVANCED_SUPERLABEL,
                )
            )
            if state.project.qwen3_target:
                items.append(
                    MenuItem("Clear custom model", clear_target_and_refresh)
                )

            # Always show rolling cont setting even though requires type 'base' and batch 1
            item = MenuItem(
                VoiceMenuShared.make_rolling_continuation_label(state.project.qwen3_rolling_cont),
                lambda _, __: VoiceMenuShared.ask_rolling_continuation(
                    state=state,
                    attribute_name="qwen3_rolling_cont",
                    max_value=Qwen3BaseModel.ROLLING_CONTINUATION_MAX_LENGTH,
                    qualifier_line="Qwen3-TTS model must be of type \"base\", and batch size must be 1."
                )
            )
            items.append(item)

            default_temp = generate_defaults.get(
                "temperature", Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT
            )
            item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="qwen3_temperature",
                default_value=default_temp,
                min_value=Qwen3BaseModel.TEMPERATURE_MIN,
                max_value=Qwen3BaseModel.TEMPERATURE_MAX
            )
            items.append(item)

            default_top_p = generate_defaults.get(
                "top_p", Qwen3BaseModel.TOP_P_DEFAULT
            )
            item = VoiceMenuShared.make_top_p_item(
                state=state,
                attr="qwen3_top_p",
                default_value=default_top_p
            )
            items.append(item)

            default_top_k = generate_defaults.get(
                "top_k", Qwen3BaseModel.TOP_K_DEFAULT
            )
            item = VoiceMenuShared.make_top_k_item(
                state=state,
                attr="qwen3_top_k",
                default_value=default_top_k
            )
            items.append(item)

            default_rp = generate_defaults.get(
                "repetition_penalty", Qwen3BaseModel.REPETITION_PENALTY_DEFAULT
            )
            item = VoiceMenuShared.make_repetition_penalty_item(
                state=state,
                attr="qwen3_repetition_penalty",
                default_value=default_rp
            )
            items.append(item)

            items.append(VoiceMenuShared.make_seed_item(state, "qwen3_seed", add_batch_warning=True))
            return items

        # TODO: not using atm; revisit, reword
        def make_subheading(_: State) -> str:
            model_type = get_model_type()
            subheading = "Qwen3-TTS supports different \"model types\".\n"
            subheading += f"The current model type, {model_type}, requires\n"
            match model_type:
                case "base":
                    subheading += "a voice clone sample.\n"
                case "custom_voice":
                    subheading += "a speaker id and an optional instruction.\n"
                case _:
                    subheading = ""
            return subheading

        VoiceMenuShared.menu_wrapper(state, make_items)

# ---

def model_target_submenu(
    state: State,
    apply_target: Callable[[str], None] | None = None,
) -> None:
    resolved_apply_target = apply_target
    if resolved_apply_target is None:
        resolved_apply_target = lambda target: apply_model_and_validate(state, target)

    VoiceMenuShared.target_submenu(
        state=state,
        heading="Select Qwen3-TTS model",
        preset_targets=Qwen3BaseModel.PRESET_REPO_IDS,
        current_target=state.project.qwen3_target,
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

    model_name = Tts.get_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"zeropointnine/Darwin-TTS-1.7B-Cross-Qwen3Tokenizer\" or \"/path/to/checkpoint\""

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.qwen3_target,
        callback=lambda _, target: resolved_apply_target(target)
    )

def apply_model_and_validate(
    state: State,
    target: str,
    on_applied: Callable[[TtsInspected], None] | None = None,
) -> None:
    project = state.project

    previous_target = project.qwen3_target
    previous_model_type = project.qwen3_model_type
    previous_speaker_id = project.qwen3_speaker_id

    def revert() -> None:
        project.qwen3_target = previous_target
        project.qwen3_model_type = previous_model_type
        project.qwen3_speaker_id = previous_speaker_id
        _ = ModelWorker.clear_models_if_running_blocking()

    project.qwen3_target = target
    _ = ModelWorker.clear_models_if_running_blocking()

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        printt()
        printt(f"{COL_ERROR}Contents at {target} appear to be invalid:")
        printt(f"{COL_ERROR}{error}")
        printt()
        revert()
        ask.ask_enter_to_continue()
        return

    metadata = inspection.metadata or {}
    inspected_type = str(metadata.get("model_type", ""))
    if not bool(metadata.get("is_model_type_supported", True)):
        print_feedback(f"Unsupported type: {inspected_type}", is_error=True)
        revert()
        ask.ask_enter_to_continue()
        return
    if project.qwen3_speaker_id:
        project.qwen3_speaker_id = ""
    project.qwen3_model_type = inspected_type
    project.save()
    if on_applied is not None:
        on_applied(inspection)
    print_feedback("Model set:", target)
    ask.ask_enter_to_continue()

def on_clear_model_target(state: State, __: MenuItem) -> None:
    state.project.qwen3_target = ""
    state.project.qwen3_model_type = ""
    state.project.save()
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared, will use default model")

def ask_speaker_id(project: Project, speakers: list[str]) -> None:
    if len(speakers) == 1:
        message = f"Model has only one speaker id ({speakers[0]})"
        print_feedback(message)
        return

    def validate_speaker_id(value: str) -> str:
        return "" if value in speakers else "Invalid speaker id"

    ask.ask_string_and_save(
        project,
        f"Choose a speaker:\n{speakers}\n",
        "qwen3_speaker_id",
        "Set speaker id:",
        validator=validate_speaker_id,
    )

def ask_instructions(project: Project) -> None:
    ask.ask_string_and_save(
        project,
        "Enter instructions prompt:",
        "qwen3_instructions",
        "Set instructions:",
    )
