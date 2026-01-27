from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import VibeVoiceProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceVibeVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        project = state.project

        def make_select_voice_label(_: State) -> str: # custom
            if not state.project.has_voice:
                if state.project.vibevoice_lora_target:
                    col = COL_ACCENT
                else:
                    col = COL_ERROR
                currently = make_currently_string("none", color_code=col)
            else:
                currently = make_currently_string(state.project.get_voice_label())
            return f"Select voice clone sample {currently}"

        def make_model_target_label(_) -> str:
            if project.vibevoice_target:
                label = make_currently_string(project.vibevoice_target)
            else:
                label = f"{COL_DIM}(optional)"
            return f"Custom model {label}"

        def make_lora_target_label(_) -> str:
            if project.vibevoice_lora_target:
                label = make_currently_string(project.vibevoice_lora_target)
            else:
                label = f"{COL_DIM}(optional)"
            return f"LoRA {label}"

        def make_cfg_label(_) -> str:
            value = make_parameter_value_string(
                project.vibevoice_cfg, VibeVoiceProtocol.CFG_DEFAULT, 1
            )
            return f"CFG scale {make_currently_string(value)}"

        def on_cfg(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                f"Enter CFG {COL_DIM}({VibeVoiceProtocol.CFG_MIN} to {VibeVoiceProtocol.CFG_MAX}):",
                VibeVoiceProtocol.CFG_MIN, VibeVoiceProtocol.CFG_MAX,
                "vibevoice_cfg",
                "CFG set to:"
            )

        def make_steps_label(_) -> str:
            value = make_parameter_value_string(
                project.vibevoice_steps, VibeVoiceProtocol.DEFAULT_NUM_STEPS, 0
            )
            return f"Steps {make_currently_string(value)}"

        def on_steps(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                "Enter num steps (1-30):",
                1, 30, # Sane range IMO
                "vibevoice_steps",
                "Num steps set to:",
                is_int=True
            )

        def on_seed(_: State, __: MenuItem) -> None:
            VoiceMenuShared.ask_seed_and_save(state, "vibevoice_seed")

        def make_items(_: State) -> list[MenuItem]:

            items = []

            # Voice
            items.append(
                MenuItem(
                    make_select_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.VIBEVOICE)
                )
            )
            if state.project.vibevoice_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.VIBEVOICE) 
                )
            
            # LoRA
            items.append(
                MenuItem(make_lora_target_label, lambda _, __: ask_lora_target(state.project))
            )
            if state.project.vibevoice_lora_target:
                items.append(MenuItem("Clear LoRA", on_clear_lora))

            # Model
            items.append(
                MenuItem(make_model_target_label, lambda _, __: ask_model_target(state.project))
            )
            if state.project.vibevoice_target:
                items.append(
                    MenuItem("Clear custom model", lambda _, __: clear_custom_model(state.project))
                )

            # Other config
            items.append(MenuItem(make_cfg_label, on_cfg))
            items.append(MenuItem(make_steps_label, on_steps))
            seed_string = str(state.project.vibevoice_seed) if state.project.vibevoice_seed != -1 else "random"
            items.append( MenuItem(make_menu_label("Seed", seed_string), on_seed))
            
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)

# ---

def ask_model_target(project: Project) -> None: 
    
    model_name = Tts.get_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"vibevoice/VibeVoice-7B\"; \"/path/to/checkpoint\""

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.qwen3_target, 
        callback=apply_model_and_validate
    )

def apply_model_and_validate(project: Project, target: str) -> None: 

    project.vibevoice_target = target
    project.save()
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model() # for good measure
    try:
        _ = Tts.get_vibevoice()
    except (OSError, Exception) as e:
        # Revert
        project.vibevoice_target = ""
        project.save()
        Tts.set_model_params_using_project(project)
        AskUtil.ask_error(f"\n{make_error_string(e)}")
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

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.qwen3_target, 
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
    try:
        instance = Tts.get_vibevoice()
    except Exception as e:
        revert()
        AskUtil.ask_error(f"\n{make_error_string(e)}")
        return

    if instance.has_lora:
        print_feedback("\nLoRA set:", target)
    else:
        revert()
        AskUtil.ask_error("\n{COL_ERROR}Couldn't load LoRA")

def on_clear_lora(state: State, __: MenuItem) -> None:
    state.project.vibevoice_lora_target = ""
    state.project.save()
    Tts.set_model_params_using_project(state.project)
    Tts.clear_tts_model()
    print_feedback("Cleared LoRA")
