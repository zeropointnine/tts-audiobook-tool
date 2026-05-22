from tts_audiobook_tool import ask
from tts_audiobook_tool.l import L
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared


class VoiceOmniVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_voice_label(_) -> str:
            if not state.project.omnivoice_voice_file_name:
                currently = make_currently_string("none", value_prefix="", color_code=COL_DIM)
            else:
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
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

        def make_speed_label(_) -> str:
            speed = state.project.omnivoice_speed
            return f"Speed {make_currently_string(speed, default=OmniVoiceBaseModel.DEFAULT_SPEED, num_decimals=1)}"
        
        def make_steps_label(_) -> str:
            steps = state.project.omnivoice_num_step
            return f"Inference steps {make_currently_string(steps, default=OmniVoiceBaseModel.DEFAULT_STEPS)}"

        def on_clear_voice(s: State, __: MenuItem) -> None:
            ProjectVoiceUtil.clear_voice_and_save(s.project, TtsModelInfos.OMNIVOICE)
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

            items.append(
                MenuItem(
                    make_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.OMNIVOICE)
                )
            )
            if state.project.omnivoice_voice_file_name:
                items.append(MenuItem("Clear voice clone sample", on_clear_voice))

            items.append(
                MenuItem(make_instruct_label, lambda _, __: ask_instruct(state.project))
            )
            if state.project.omnivoice_instruct:
                items.append(MenuItem("Clear instructions", on_clear_instruct))

            items.append(
                MenuItem(make_target_label, lambda _, __: ask_target(state.project))
            )
            if state.project.omnivoice_target:
                items.append(MenuItem("Clear custom model", on_clear_model_target))

            steps_item = MenuItem(make_steps_label, lambda _, __: ask_steps(state.project))
            steps_item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(steps_item)

            speed_item = MenuItem(make_speed_label, lambda _, __: ask_speed(state.project))
            items.append(speed_item)

            cfg_item = MenuUtil.make_number_item(
                state=state,
                attr="omnivoice_cfg",
                base_label="CFG",
                default_value=OmniVoiceBaseModel.CFG_DEFAULT,
                is_minus_one_default=True,
                num_decimals=2,
                prompt=f"Enter CFG",
                min_value=OmniVoiceBaseModel.CFG_MIN,
                max_value=OmniVoiceBaseModel.CFG_MAX
            )
            items.append(cfg_item)

            items.append(
                VoiceMenuShared.make_seed_item(state, "omnivoice_seed")
            )

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_instruct(project: Project) -> None:
    printt("Enter voice design instructions:")
    printt(f"{COL_DIM}Eg: \"male, british accent, low pitch\" / \"female, young adult, high pitch\"")
    if project.omnivoice_voice_file_name:
        printt(f"{COL_DIM}Note: When used alongside voice cloning, instructions may have minimal effect")
    inp = ask.ask(lower=False)
    if not inp:
        return

    error, normalized = validate_instruct(inp)
    if error:
        print_feedback(error, is_error=True)
        return

    project.omnivoice_instruct = normalized
    print_feedback("Set instructions:", truncate_pretty(normalized, 60))


def validate_instruct(instruct: str) -> tuple[str, str]:
    """
    Best-effort pre-validation using OmniVoice's own instruct resolver, so users
    get feedback at menu time instead of waiting for inference.

    Returns:
        (error_message, normalized_instruct)

    If OmniVoice isn't importable in the current environment, validation is
    skipped and the original value is returned unchanged.
    """
    try:
        from omnivoice.models.omnivoice import _resolve_instruct  # type: ignore
    except Exception as e:
        L.e(f"{instruct} - {e}")
        return "", instruct

    try:
        normalized = _resolve_instruct(instruct)
    except ValueError as e:
        return make_error_string(e), ""
    except Exception:
        # Can't validate, just allow
        return "", instruct

    if normalized is None:
        return "", ""
    return "", normalized


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
    previous_target = project.omnivoice_target

    def revert() -> None:
        project.omnivoice_target = previous_target
        project.save()
        Tts.set_model_params_using_project(project)
        Tts.clear_tts_model()

    project.omnivoice_target = target
    Tts.set_model_params_using_project(project)
    Tts.clear_tts_model() # for good measure

    try:
        Tts.get_omnivoice()
    except Exception as e:
        revert()
        print_feedback(f"Failed to load OmniVoice model: {e}", is_error=True)
        return

    project.save()
    print_feedback("Model set:", target)


def ask_speed(project: Project) -> None:
    prompt = f"Enter speech speed {COL_DIM}(0.5–2.0; default 1.0; or -1 to reset){COL_DEFAULT}: "
    inp = ask.ask(prompt, lower=False)
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


def ask_steps(project: Project) -> None:
    s = (
        f"Enter number of inference steps: {COL_DIM}"
        f"({OmniVoiceBaseModel.MIN_STEPS}–{OmniVoiceBaseModel.MAX_STEPS}; "
        f"default {OmniVoiceBaseModel.DEFAULT_STEPS}) "
        f"\nSmaller values = faster, reduced quality"
    )
    printt(s)
    inp = ask.ask()
    if not inp:
        return
    try:
        value = int(inp)
    except ValueError:
        print_feedback("Invalid value", is_error=True)
        return
    if value != -1 and not (OmniVoiceBaseModel.MIN_STEPS <= value <= OmniVoiceBaseModel.MAX_STEPS):
        print_feedback(
            f"Value must be between {OmniVoiceBaseModel.MIN_STEPS} and {OmniVoiceBaseModel.MAX_STEPS} (or -1 to reset)",
            is_error=True
        )
        return
    project.omnivoice_num_step = value
    print_feedback("Inference steps set:", str(value) if value != -1 else f"default ({OmniVoiceBaseModel.DEFAULT_STEPS})")