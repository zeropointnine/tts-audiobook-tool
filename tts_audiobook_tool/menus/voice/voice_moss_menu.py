from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
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
            value = state.project.moss_target or MossConfigs.get_default_repo_id()
            value = value.removeprefix("OpenMOSS-Team/")
            value = ellipsize_path_for_menu(value)
            label = make_currently_string(value, default=MossConfigs.get_default_repo_id())
            return f"Select MOSS-TTS model {label}"

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

    def make_preset_label(target: str) -> str:
        label = target
        if target == MossConfigs.get_default_repo_id():
            label += f" {COL_DIM}(default)"
        if target == state.project.moss_target:
            label += f" {COL_ACCENT}(selected)"
        return label

    items = []
    for config in list(MossConfigs):
        target = config.value.repo_id
        items.append(MenuItem(make_preset_label(target), lambda _, __, target=target: apply_model_and_validate(state.project, target)))
        items[-1].sublabel = config.preset_description
    items.append(MenuItem("Enter custom hf repo id or local path", lambda _, __: ask_target(state.project)))

    MenuUtil.menu(
        state=state,
        heading="Select MOSS-TTS model",
        items=items,
        one_shot=True
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
