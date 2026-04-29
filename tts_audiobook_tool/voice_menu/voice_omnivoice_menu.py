from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared


class VoiceOmniVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_voice_label(_) -> str:
            if not state.project.omnivoice_voice_file_name:
                currently = make_currently_string("none", value_prefix="", color_code=COL_DIM)
            else:
                currently = make_currently_string(state.project.voice_label)
            return f"Select voice clone sample {currently}"

        def make_instruct_label(_) -> str:
            if not state.project.omnivoice_instruct:
                suffix = f"{COL_DIM}(optional)"
            else:
                value = truncate_pretty(state.project.omnivoice_instruct, 40, content_color=COL_ACCENT)
                suffix = make_currently_string(value)
            return f"Voice design instructions {suffix}"

        def make_target_label(_) -> str:
            target = state.project.omnivoice_target or OmniVoiceBaseModel.DEFAULT_REPO_ID
            is_default = (target == OmniVoiceBaseModel.DEFAULT_REPO_ID)
            if is_default:
                label = f"{COL_DIM}(optional)"
            else:
                value = ellipsize_path_for_menu(target)
                label = make_currently_string(value)
            return f"Custom model {label}"

        def make_dtype_label(_) -> str:
            dtype = state.project.omnivoice_dtype or "float16"
            return f"Precision {make_currently_string(dtype)}"

        def make_speed_label(_) -> str:
            speed = state.project.omnivoice_speed
            value = str(speed) if speed != -1 else f"1.0 {COL_DIM}(default)"
            return f"Speed {make_currently_string(value)}"
        
        def make_num_step_label(_) -> str:
            num_step = state.project.omnivoice_num_step
            value = str(num_step) if num_step != -1 else f"32 {COL_DIM}(default)"
            return f"Diffusion steps {make_currently_string(value)}"

        def on_clear_voice(s: State, __: MenuItem) -> None:
            s.project.clear_voice_and_save(TtsModelInfos.OMNIVOICE)
            print_feedback("Voice clone cleared")

        def on_clear_instruct(s: State, __: MenuItem) -> None:
            s.project.omnivoice_instruct = ""
            print_feedback("Instructions cleared")

        def on_clear_model_target(s: State, __: MenuItem) -> None:
            s.project.omnivoice_target = ""
            Tts.set_model_params_using_project(s.project)
            Tts.clear_tts_model()
            print_feedback("Cleared, will use default model")


        def make_items(_: State) -> list[MenuItem]:
            items = []

            # ── Voice clone ──────────────────────────────────────────────
            items.append(
                MenuItem(
                    make_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.OMNIVOICE)
                )
            )
            if state.project.omnivoice_voice_file_name:
                items.append(MenuItem("Clear voice clone sample", on_clear_voice))

            # ── Voice design instructions ────────────────────────────────
            items.append(
                MenuItem(make_instruct_label, lambda _, __: ask_instruct(state.project))
            )
            if state.project.omnivoice_instruct:
                items.append(MenuItem("Clear instructions", on_clear_instruct))

            # ── Custom model ─────────────────────────────────────────────
            items.append(
                MenuItem(make_target_label, lambda _, __: ask_target(state.project))
            )
            if state.project.omnivoice_target:
                items.append(MenuItem("Clear custom model", on_clear_model_target))

            # ── Advanced params ──────────────────────────────────────────
            items.append(
                MenuItem(
                    make_dtype_label,
                    lambda _, __: ask_dtype(state.project),
                    superlabel=VOICE_ADVANCED_SUPERLABEL
                )
            )
            items.append(
                MenuItem(make_speed_label, lambda _, __: ask_speed(state.project))
            )
            items.append(
                MenuItem(make_num_step_label, lambda _, __: ask_num_step(state.project))
            )
            items.append(
                VoiceMenuShared.make_seed_item(state, "omnivoice_seed")
            )

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_instruct(project: Project) -> None:
    printt("Enter voice design instructions:")
    printt(f"{COL_DIM}Eg: \"male, deep voice, british accent\" / \"female, young adult, high pitch\"")
    inp = AskUtil.ask(lower=False)
    if not inp:
        return
    project.omnivoice_instruct = inp
    print_feedback("Set instructions:", truncate_pretty(inp, 60))


def ask_target(project: Project) -> None:
    model_name = Tts.get_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"k2-fsa/OmniVoice\" or \"/path/to/local/checkpoint\""
    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.omnivoice_target,
        callback=apply_target
    )


def apply_target(project: Project, target: str) -> None:
    project.omnivoice_target = target
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model()
    print_feedback("Model set:", target)


def ask_dtype(project: Project) -> None:
    options = ["float16", "bfloat16", "float32"]
    prompt = f"Select precision {COL_DIM}(float16 recommended for CUDA; float32 for CPU/MPS){COL_DEFAULT}"
    prompt += f"\nOptions: {options}"
    inp = AskUtil.ask(prompt, lower=True)
    if not inp:
        return
    if inp not in options:
        print_feedback(f"Invalid option. Choose from: {options}", is_error=True)
        return
    project.omnivoice_dtype = inp
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model()
    print_feedback("Precision set:", inp)


def ask_speed(project: Project) -> None:
    prompt = f"Enter speech speed {COL_DIM}(0.5–2.0; default 1.0; or -1 to reset){COL_DEFAULT}: "
    inp = AskUtil.ask(prompt, lower=False)
    if not inp:
        return
    try:
        value = float(inp)
    except ValueError:
        print_feedback("Invalid value", is_error=True)
        return
    if value != -1 and not (0.5 <= value <= 2.0):
        print_feedback("Value must be between 0.5 and 2.0 (or -1 to reset)", is_error=True)
        return
    project.omnivoice_speed = value
    print_feedback("Speed set:", str(value) if value != -1 else "default (1.0)")


def ask_num_step(project: Project) -> None:
    prompt = f"Enter number of diffusion steps {COL_DIM}(1–128; default 32; use 16 for ~2x faster){COL_DEFAULT}: "
    inp = AskUtil.ask(prompt, lower=False)
    if not inp:
        return
    try:
        value = int(inp)
    except ValueError:
        print_feedback("Invalid value", is_error=True)
        return
    if value != -1 and not (1 <= value <= 128):
        print_feedback("Value must be between 1 and 128 (or -1 to reset)", is_error=True)
        return
    project.omnivoice_num_step = value
    print_feedback("Diffusion steps set:", str(value) if value != -1 else "default (32)")