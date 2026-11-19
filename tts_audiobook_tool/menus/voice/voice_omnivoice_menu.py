from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DIM, COL_DIM_ITALICS, ellipsize_path_for_menu, make_currently_string, print_feedback, printt
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

MODEL_ID = "omnivoice_local"


class VoiceOmniVoiceMenu:
    """The shared voice-sample items plus an optional LoRA (see `omnivoice_lora`),
    following the VibeVoice LoRA interaction."""

    @staticmethod
    def menu(state: State) -> None:
        project = state.project

        def make_lora_target_label(_) -> str:
            if project.get_model_setting(MODEL_ID, 'lora_target'):
                value = ellipsize_path_for_menu(project.get_model_setting(MODEL_ID, 'lora_target'))
                return f"LoRA {make_currently_string(value)}"
            return f"Select LoRA {COL_DIM}(optional)"

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id(MODEL_ID))
            items.append(
                MenuItem(make_lora_target_label, lambda _, __: ask_lora_target(state), blank_line_before=True)
            )
            if project.get_model_setting(MODEL_ID, 'lora_target'):
                items.append(MenuItem("Clear LoRA", on_clear_lora))
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


def ask_lora_target(state: State) -> None:
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    project = state.project
    current = project.get_model_setting(MODEL_ID, 'lora_target')

    prompt = "Enter huggingface repo id or local directory path to OmniVoice LoRA adapter"
    prompt += f"\n{COL_DIM}Eg, \"My-Weird-Prompts/Herman-LoRA-V3\", \"/path/to/checkpoint-900\"."
    prompt += f"\n{COL_DIM}For a hf subdirectory, enter, for example: \"owner/repo/sub/dir\"."
    prompt += f"\n{COL_DIM}A LoRA trained on one speaker works without a voice sample."

    ModelMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=current,
        callback=lambda _, target: apply_lora_and_validate(state, target)
    )


def apply_lora_and_validate(state: State, target: str) -> None:
    project = state.project

    previous_target = project.get_model_setting(MODEL_ID, 'lora_target')

    def revert() -> None:
        project.set_model_setting(MODEL_ID, 'lora_target', previous_target)
        _ = ModelWorker.clear_models_if_running_blocking()

    project.set_model_setting(MODEL_ID, 'lora_target', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"{error}")
        return

    metadata = inspection.metadata or {}
    if bool(metadata.get("has_lora", False)):
        project.save()
        print_feedback("LoRA set:", target)
        ask.ask_enter_to_continue()
    else:
        revert()
        reason = metadata.get("lora_error") or ""
        ask.ask_error(f"Couldn't load LoRA: {reason}" if reason else "Couldn't load LoRA")


def on_clear_lora(state: State, __: MenuItem) -> None:
    state.project.set_model_setting(MODEL_ID, 'lora_target', "")
    state.project.save()
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared LoRA")
