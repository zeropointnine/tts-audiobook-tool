from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import Qwen3Protocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceQwen3Menu:
    """
    Note, unlike the other model voice/settings menus, 
    this will instantiate model by necessity if not already
    """

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        def make_voice_label(_) -> str:
            if not state.project.qwen3_voice_file_name:
                currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
            else:
                currently = make_currently_string(state.project.get_voice_label())
            return f"Select voice clone sample {currently}"

        def make_target_label(_) -> str:
            target = state.project.qwen3_target or Qwen3Protocol.DEFAULT_REPO_ID
            is_default = (target == Qwen3Protocol.DEFAULT_REPO_ID)
            if is_default:
                label = f"{COL_DIM}(optional)"
            else:
                value = truncate_path_pretty(target)
                label = make_currently_string(value)
                label += f" {COL_DIM}(model type: {Tts.get_qwen3().model_type})"
            return f"Custom model {label}"

        def make_speaker_label(_) -> str:
            speakers = Tts.get_qwen3().supported_speakers
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

        def make_temperature_label(_) -> str:
            default = Tts.get_qwen3().generate_defaults.get(
                "temperature", Qwen3Protocol.TEMPERATURE_FALLBACK_DEFAULT
            )
            value = make_parameter_value_string(
                state.project.qwen3_temperature, default, 2
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                state.project,
                f"Enter temperature ({Qwen3Protocol.TEMPERATURE_MIN} to {Qwen3Protocol.TEMPERATURE_MAX}):",
                Qwen3Protocol.TEMPERATURE_MIN, Qwen3Protocol.TEMPERATURE_MAX, 
                "qwen3_temperature",
                "Temperature set to:"
            )

        def on_seed(_: State, __: MenuItem) -> None:
            VoiceMenuShared.ask_seed_and_save(state, "qwen3_seed")


        def make_items(_: State) -> list[MenuItem]:
            
            items = []
            
            match Tts.get_qwen3().model_type:
                case "base":
                    # Voice clone, clear voice clone
                    items.append(
                        MenuItem(
                            make_voice_label,
                            lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.QWEN3TTS)
                        )                
                    )
                    if state.project.qwen3_voice_file_name:
                        items.append( VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.QWEN3TTS))
                case "custom_voice":
                    # Speaker id, instructions
                    items.append(
                        MenuItem(make_speaker_label, lambda _, __: ask_speaker_id(state.project))
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
                MenuItem(make_target_label, lambda _, __: ask_target(state.project))
            )
            if state.project.qwen3_target:
                items.append(
                    MenuItem("Clear custom model", on_clear_model_target)
                )
            
            # Other params
            items.append(
                MenuItem(make_temperature_label, on_temperature)
            )
            seed_string = str(state.project.qwen3_seed) if state.project.qwen3_seed != -1 else "random"
            items.append( 
                MenuItem(make_menu_label("Seed", seed_string), on_seed)
            )
            return items
        
        # TODO: not using atm; revisit, reword
        def make_subheading(_: State) -> str:
            model_type = Tts.get_qwen3().model_type
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

        VoiceMenuShared.show_voice_menu(state, make_items)

# ---

def ask_target(project: Project) -> None: 

    model_name = Tts.get_type().value.ui["short_name"],
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\"; \"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign\"; \"/path/to/checkpoint\""

    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.qwen3_target, 
        callback=apply_model_and_validate
    )
    
def apply_model_and_validate(project: Project, target: str) -> None: 

    def revert() -> None:
        project.qwen3_target = ""
        project.save()
        Tts.set_model_params_using_project(project)
        Tts.clear_tts_model()

    project.qwen3_target = target
    project.save()

    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model() # for good measure

    # Instantiate model with new settings and check if valid
    try:
        instance = Tts.get_qwen3()
        if not instance.is_model_type_supported:
            print_feedback(f"Unsupported type: {instance.model_type}", is_error=True)
            revert()
            AskUtil.ask_enter_to_continue()
            return        
        
        # Success
        if project.qwen3_speaker_id:
            # Invalidate speaker id (but keep instructions ig)
            project.qwen3_speaker_id = ""
            project.save()
        print_feedback("Model set:", target)

    except (OSError, Exception) as e:
        printt()
        printt(f"{COL_ERROR}Contents at {target} appear to be invalid:")
        printt(f"{COL_ERROR}{make_error_string(e)}")
        printt()
        revert()
        AskUtil.ask_enter_to_continue()

def on_clear_model_target(state: State, __: MenuItem) -> None:
    state.project.qwen3_target = ""
    state.project.save()
    Tts.set_model_params_using_project(state.project)
    Tts.clear_tts_model()
    print_feedback("Cleared, will use default model")
    # Preemptively re-instantiate model (can't be helped)
    _ = Tts.get_qwen3()

def ask_speaker_id(project: Project) -> None:
    speakers = Tts.get_qwen3().supported_speakers
    if len(speakers) == 1:
        message = "Model has only one speaker id ({speakers[0]})"
        print_feedback(message)
        return
    prompt = f"Choose a speaker:\n{speakers}\n"
    inp = AskUtil.ask(prompt, lower=False)
    if not inp:
        return
    if not inp in speakers:
        print_feedback("Invalid speaker id", is_error=True)
        return
    project.qwen3_speaker_id = inp
    project.save()
    print_feedback("Set speaker id:", inp)

def ask_instructions(project: Project) -> None:
    printt("Enter instructions prompt:")
    inp = AskUtil.ask(lower=False)
    if not inp:
        return
    project.qwen3_instructions = inp
    project.save()
    print_feedback("Set instructions: ", truncate_pretty(inp, 60))
