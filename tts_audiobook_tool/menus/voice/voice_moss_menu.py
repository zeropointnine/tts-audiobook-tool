from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs, MossBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceMossMenu:
    """
    MOSS model settings menu
    Is used for both MOSS and SERVER_MOSS
    """

    @staticmethod
    def menu(state: State) -> None:

        def make_target_label(_) -> str:
            return VoiceMenuShared.make_target_label(
                label_prefix="Select MOSS-TTS model",
                target=state.project.moss_target,
                default_target=MossConfigs.get_default_repo_id(),
                remove_prefixes=["OpenMOSS-Team/"],
            )

        def make_items(_: State) -> list[MenuItem]:

            items = []

            VoiceMossShared.append_voice_items(items, state, TtsModelType.MOSS)

            items.append(
                MenuItem(
                    make_target_label, 
                    lambda _, __: target_submenu(state), 
                    superlabel=VOICE_ADVANCED_SUPERLABEL
                )
            )

            item = MenuItem(
                VoiceMenuShared.make_rolling_continuation_label(state.project.moss_rolling_cont),
                lambda _, __: VoiceMenuShared.ask_rolling_continuation(
                    state=state,
                    attribute_name="moss_rolling_cont",
                    max_value=MossBaseModel.ROLLING_CONTINUATION_MAX_LENGTH,
                    qualifier_line="MOSS-TTS rolling continuation requires batch size 1."
                )
            )
            items.append(item)

            config = MossConfigs.get_by_target(state.project.moss_target)

            items.append(VoiceMossShared.make_temperature_item(state, config))

            items.append(VoiceMossShared.make_audio_top_p_item(state, config))

            items.append(VoiceMossShared.make_audio_top_k_item(state, config))

            item = VoiceMenuShared.make_seed_item(state, "moss_seed", add_batch_warning=True)
            items.append(item)

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)

# ---

def target_submenu(state: State) -> None:
    configs = list(MossConfigs)
    VoiceMenuShared.target_submenu(
        state=state,
        heading="Select MOSS-TTS model",
        preset_targets=[config.value.repo_id for config in configs],
        current_target=state.project.moss_target,
        default_target=MossConfigs.get_default_repo_id(),
        ask_custom_target=lambda: ask_target(state),
        apply_target=lambda target: apply_model_and_validate(state, target),
        sublabels=[config.preset_description for config in configs],
    )

def ask_target(state: State) -> None:
    project = state.project

    model_name = Tts.get_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"OpenMOSS-Team/MOSS-TTS-v1.5\" or \"/path/to/checkpoint\""

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.moss_target,
        callback=lambda _, target: apply_model_and_validate(state, target)
    )

def apply_model_and_validate(state: State, target: str) -> None:
    project = state.project

    previous_target = project.moss_target

    def revert() -> None:
        project.moss_target = previous_target
        _ = ModelWorker.clear_models_if_running_blocking()

    project.moss_target = target
    _ = ModelWorker.clear_models_if_running_blocking()

    # Preset targets are pinned, known-good repos whose runtime properties
    # (arch, sample rate, sampling defaults) are fully described by
    # MossConfigs, so there is nothing to learn from loading the model here.
    # Custom targets run arbitrary remote code, so validate those by loading.
    if MossConfigs.get_preset_by_target(target) is not None:
        project.save()
        print_feedback("Model set:", target)
        return

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"\nContents at {target} appear to be invalid:\n{error}")
        return

    project.save()
    print_feedback("Model set:", target)
    ask.ask_enter_to_continue()

def on_clear_model_target(state: State, __: MenuItem) -> None:
    state.project.moss_target = ""
    state.project.save()
    _ = ModelWorker.clear_models_if_running_blocking()
    print_feedback("Cleared, will use default model")
