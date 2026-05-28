from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs, MossVoiceCloneMode
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceMossMenu:

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
            items.append(
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.MOSS)
                )
            )
            if state.project.moss_voice_file_name:
                items.append(VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.MOSS))

            items.append(
                MenuItem(make_target_label, lambda _, __: target_submenu(state))
            )
            if state.project.moss_target:
                items.append(
                    MenuItem("Clear custom model", on_clear_model_target)
                )

            item = MenuItem(
                make_voice_clone_mode_label,
                lambda _, __: VoiceMossMenu.voice_clone_mode_menu(state)
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            config = MossConfigs.get_by_target(state.project.moss_target)
            items.append(make_temperature_item(state, config))
            items.append(make_audio_top_p_item(state, config))
            items.append(make_audio_top_k_item(state, config))

            prompt = f"Enter a static seed value {COL_DIM}(or -1 for random){COL_DEFAULT}"
            prompt += f"\n{COL_DIM}(Note, audio generations are not idempotent when using batch mode): "
            items.append(VoiceMenuShared.make_seed_item(state, "moss_seed", prompt_override=prompt))

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def voice_clone_mode_menu(state: State) -> None:
        voice_clone_modes = list(MossVoiceCloneMode)

        def on_select(value: MossVoiceCloneMode) -> None:
            if state.project.moss_mode != value:
                state.project.moss_mode = value
                state.project.save()
            print_feedback("Set to:", value.label)

        MenuUtil.options_menu(
            state=state,
            heading_text="Voice clone mode",
            labels=[mode.label for mode in voice_clone_modes],
            sublabels=[mode.description for mode in voice_clone_modes],
            values=voice_clone_modes,
            current_value=state.project.moss_mode,
            default_value=MossVoiceCloneMode.get_default(),
            on_select=on_select,
        )

# ---

def make_voice_clone_mode_label(state: State) -> str:
    value = MossVoiceCloneMode.normalize(state.project.moss_mode).id
    default_value = MossVoiceCloneMode.get_default().id
    return make_menu_label("Voice clone mode", value, default_value)

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

def get_temperature_attr(arch_type: MossConfigs) -> str:
    return "moss_local_temperature" if arch_type == MossConfigs.LOCAL else "moss_delay_temperature"

def get_top_p_attr(arch_type: MossConfigs) -> str:
    return "moss_local_top_p" if arch_type == MossConfigs.LOCAL else "moss_delay_top_p"

def get_top_k_attr(arch_type: MossConfigs) -> str:
    return "moss_local_top_k" if arch_type == MossConfigs.LOCAL else "moss_delay_top_k"

def make_temperature_item(state: State, arch_type: MossConfigs) -> MenuItem:
    arch_values = arch_type.value
    return VoiceMenuShared.make_temperature_item(
        state=state,
        attr=get_temperature_attr(arch_type),
        base_label=f"{arch_values.arch_name} temperature",
        default_value=arch_values.temperature_default,
        min_value=arch_values.temperature_min,
        max_value=arch_values.temperature_max,
    )

def make_audio_top_p_item(state: State, arch_type: MossConfigs) -> MenuItem:

    arch_values = arch_type.value

    return MenuUtil.make_number_item(
        state=state,
        attr=get_top_p_attr(arch_type),
        base_label=f"{arch_values.arch_name} audio top-p",
        default_value=arch_values.top_p_default,
        is_minus_one_default=True,
        num_decimals=2,
        prompt=f"Enter Audio top-p {COL_DIM}({arch_values.top_p_min} to {arch_values.top_p_max}){COL_DEFAULT}:",
        min_value=arch_values.top_p_min,
        max_value=arch_values.top_p_max
    )

def make_audio_top_k_item(state: State, arch_type: MossConfigs) -> MenuItem:

    arch_values = arch_type.value

    return MenuUtil.make_number_item(
        state=state,
        attr=get_top_k_attr(arch_type),
        base_label=f"{arch_values.arch_name} audio top-k",
        default_value=arch_values.top_k_default,
        is_minus_one_default=True,
        num_decimals=0,
        prompt=f"Enter Audio top-K {COL_DIM}({arch_values.top_k_min} to {arch_values.top_k_max}){COL_DEFAULT}:",
        min_value=arch_values.top_k_min,
        max_value=arch_values.top_k_max
    )
