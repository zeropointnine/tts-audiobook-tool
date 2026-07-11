from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceVibeVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        project = state.project

        def make_select_voice_label(_: State) -> str: # custom
            if not ProjectVoiceUtil.has_voice(state.project):
                if state.project.vibevoice_lora_target:
                    col = COL_ACCENT
                else:
                    col = COL_ERROR
                currently = make_currently_string("none", color_code=col)
            else:
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
            return f"Select voice clone sample {currently}"

        def make_model_target_label(_) -> str:
            return VoiceMenuShared.make_target_label(
                label_prefix="Select model",
                target=project.vibevoice_target,
                default_target=VibeVoiceBaseModel.DEFAULT_REPO_ID,
            )

        def make_lora_target_label(_) -> str:
            if project.vibevoice_lora_target:
                value = ellipsize_path_for_menu(project.vibevoice_lora_target)
                label = make_currently_string(value)
            else:
                label = f"{COL_DIM}(optional)"
            return f"Select LoRA {label}"

        def make_items(_: State) -> list[MenuItem]:

            items = []

            # Voice
            items.append(
                VoiceMenuShared.make_manage_voice_samples_item(
                    state,
                    TtsModelType.VIBEVOICE,
                    no_samples_label=make_select_voice_label,
                )
            )
             
            # LoRA
            items.append(
                MenuItem(make_lora_target_label, lambda _, __: ask_lora_target(state.project))
            )
            if state.project.vibevoice_lora_target:
                items.append(MenuItem("Clear LoRA", on_clear_lora))

            # Model
            items.append(
                MenuItem(make_model_target_label, lambda _, __: target_submenu(state))
            )
            if state.project.vibevoice_target:
                items.append(
                    MenuItem("Clear custom model", lambda _, __: clear_custom_model(state.project))
                )

            # Other config
            item = MenuUtil.make_number_item(
                state=state,
                attr="vibevoice_cfg",
                base_label="CFG", 
                default_value=VibeVoiceBaseModel.CFG_DEFAULT,
                is_minus_one_default=True,
                num_decimals=2,
                prompt=f"Enter CFG {COL_DIM}({VibeVoiceBaseModel.CFG_MIN} to {VibeVoiceBaseModel.CFG_MAX}):",
                min_value=VibeVoiceBaseModel.CFG_MIN,
                max_value=VibeVoiceBaseModel.CFG_MAX
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            items.append(
                MenuUtil.make_number_item(
                    state=state,
                    attr="vibevoice_steps",
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
                VoiceMenuShared.make_seed_item(state, "vibevoice_seed")
            )
            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)

# ---

def target_submenu(state: State) -> None:
    VoiceMenuShared.target_submenu(
        state=state,
        heading="Select VibeVoice model",
        preset_targets=VibeVoiceBaseModel.PRESET_REPO_IDS,
        current_target=state.project.vibevoice_target,
        default_target=VibeVoiceBaseModel.DEFAULT_REPO_ID,
        ask_custom_target=lambda: ask_model_target(state.project),
        apply_target=lambda target: apply_model_and_validate(state.project, target),
    )

def ask_model_target(project: Project) -> None: 

    model_name = Tts.get_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"vibevoice/VibeVoice-7B\"; \"/path/to/checkpoint\""
    if project.vibevoice_target:
        prompt += f"\n{COL_DIM}(currently: {project.vibevoice_target})"    

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.vibevoice_target, 
        callback=apply_model_and_validate
    )

def apply_model_and_validate(project: Project, target: str) -> None: 

    project.vibevoice_target = target
    project.save()
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model() # for good measure

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    try:
        _ = Tts.get_vibevoice()
    except (OSError, Exception) as e:
        # Revert
        project.vibevoice_target = ""
        project.save()
        Tts.set_model_params_using_project(project)
        ask.ask_error(f"\n{make_error_string(e)}")
        return

    print_feedback("\nCustom model set:", target)

def clear_custom_model(project: Project) -> None:
    project.vibevoice_target = ""
    project.save()
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model()
    print_feedback("Cleared, will use default model")

def ask_lora_target(project: Project) -> None: 
    
    prompt = f"Enter huggingface repo id or local directory path to VibeVoice LoRA"
    prompt += f"\n{COL_DIM}Eg, \"vibevoice-community/klett\", \"/path/to/checkpoint\""
    if project.vibevoice_lora_target:
        prompt += f"\n{COL_DIM}(Currently: {project.vibevoice_lora_target})"

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.vibevoice_lora_target, 
        callback=apply_lora_and_validate
    )

def apply_lora_and_validate(project: Project, target: str) -> None: 

    def revert() -> None:
        project.vibevoice_lora_target = ""
        project.save()
        Tts.set_model_params_using_project(project)

    project.vibevoice_lora_target = target
    project.save()
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model() # for good measure

    printt(f"{COL_DIM_ITALICS}Initializing model...")
    printt()

    try:
        instance = Tts.get_vibevoice()
    except Exception as e:
        revert()
        ask.ask_error(f"\n{make_error_string(e)}")
        return

    if instance.has_lora:
        print_feedback("\nLoRA set:", target)
        ask.ask_enter_to_continue()
    else:
        revert()
        ask.ask_error("\n{COL_ERROR}Couldn't load LoRA")

def on_clear_lora(state: State, __: MenuItem) -> None:
    state.project.vibevoice_lora_target = ""
    state.project.save()
    Tts.set_model_params_using_project(state.project)
    Tts.clear_tts_model()
    print_feedback("Cleared LoRA")
