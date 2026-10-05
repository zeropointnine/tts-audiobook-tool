from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_ACCENT, COL_DIM, COL_DIM_ITALICS, COL_ERROR, ellipsize_path_for_menu, make_currently_string, print_feedback, printt
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceVibeVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:
        project = state.project

        def make_select_voice_label(_: State) -> str: # custom
            if not ProjectVoiceUtil.has_voice(state.project):
                if state.project.get_model_setting('vibevoice_local', 'lora_target'):
                    col = COL_ACCENT
                else:
                    col = COL_ERROR
                currently = make_currently_string("none", color_code=col)
            else:
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
            return f"Select voice clone sample {currently}"

        def make_lora_target_label(_) -> str:
            if project.get_model_setting('vibevoice_local', 'lora_target'):
                value = ellipsize_path_for_menu(project.get_model_setting('vibevoice_local', 'lora_target'))
                label = make_currently_string(value)
            else:
                label = f"{COL_DIM}(optional)"
            return f"Select LoRA {label}"

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("vibevoice_local"),
                no_samples_label=make_select_voice_label,
            )
            items.append(
                MenuItem(make_lora_target_label, lambda _, __: ask_lora_target(state), blank_line_before=True)
            )
            if project.get_model_setting('vibevoice_local', 'lora_target'):
                items.append(MenuItem("Clear LoRA", on_clear_lora))
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


def ask_lora_target(state: State) -> None:
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    project = state.project

    prompt = f"Enter huggingface repo id or local directory path to VibeVoice LoRA"
    prompt += f"\n{COL_DIM}Eg, \"vibevoice-community/klett\", \"/path/to/checkpoint\""
    if project.get_model_setting('vibevoice_local', 'lora_target'):
        prompt += f"\n{COL_DIM}(Currently: {project.get_model_setting('vibevoice_local', 'lora_target')})"

    ModelMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.get_model_setting('vibevoice_local', 'lora_target'),
        callback=lambda _, target: apply_lora_and_validate(state, target)
    )


def apply_lora_and_validate(state: State, target: str) -> None:
    project = state.project

    previous_target = project.get_model_setting('vibevoice_local', 'lora_target')

    def revert() -> None:
        project.set_model_setting('vibevoice_local', 'lora_target', previous_target)
        _ = ModelWorker.clear_models_if_running_blocking()

    project.set_model_setting('vibevoice_local', 'lora_target', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"{error}")
        return

    if bool((inspection.metadata or {}).get("has_lora", False)):
        project.save()
        print_feedback("LoRA set:", target)
        ask.ask_enter_to_continue()
    else:
        revert()
        ask.ask_error("Couldn't load LoRA")


def on_clear_lora(state: State, __: MenuItem) -> None:
    state.project.set_model_setting('vibevoice_local', 'lora_target', "")
    state.project.save()
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared LoRA")
