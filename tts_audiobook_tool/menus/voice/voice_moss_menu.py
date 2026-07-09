from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs, MossBaseModel
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

            VoiceMossShared.append_voice_items(items, state)

            items.append(
                MenuItem(make_target_label, lambda _, __: target_submenu(state))
            )

            item = MenuItem(
                VoiceMenuShared.make_rolling_continuation_label(state.project.moss_rolling_cont),
                lambda _, __: VoiceMenuShared.ask_rolling_continuation(
                    state=state,
                    attribute_name="moss_rolling_cont",
                    max_value=MossBaseModel.ROLLING_CONTINUATION_MAX_LENGTH,
                    qualifier_line="MOSS-TTS rolling continuation requires batch size 1."
                ),
                superlabel=VOICE_ADVANCED_SUPERLABEL
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
        ask_custom_target=lambda: ask_target(state.project),
        apply_target=lambda target: apply_model_and_validate(state.project, target),
        sublabels=[config.preset_description for config in configs],
    )

def ask_target(project: Project) -> None:

    model_name = Tts.get_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"OpenMOSS-Team/MOSS-TTS-v1.5\" or \"/path/to/checkpoint\""

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.moss_target,
        callback=apply_model_and_validate
    )

def apply_model_and_validate(project: Project, target: str) -> None:

    previous_target = project.moss_target

    def revert() -> None:
        project.moss_target = previous_target
        project.save()
        Tts.set_model_params_using_project(project)
        Tts.clear_tts_model()

    project.moss_target = target
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model()

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    try:
        _ = Tts.get_moss()
    except Exception as e:
        printt()
        printt(f"{COL_ERROR}Contents at {target} appear to be invalid:")
        printt(f"{COL_ERROR}{make_error_string(e)}")
        printt()
        revert()
        ask.ask_enter_to_continue()
        return

    project.save()
    print_feedback("Model set:", target)
    ask.ask_enter_to_continue()

def on_clear_model_target(state: State, __: MenuItem) -> None:
    state.project.moss_target = ""
    state.project.save()
    Tts.set_model_params_using_project(state.project)
    Tts.clear_tts_model()
    print_feedback("Cleared, will use default model")
