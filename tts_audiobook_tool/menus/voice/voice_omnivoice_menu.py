from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared


class VoiceOmniVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_voice_label(_) -> str:
            if not state.project.get_model_setting('omnivoice_local', 'file_name'):
                currently = make_currently_string("none", value_prefix="", color_code=COL_ERROR)
            else:
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
            return f"Select voice clone sample {currently}"

        def make_target_label(_) -> str:
            target = state.project.get_model_setting('omnivoice_local', 'target') or OmniVoiceBaseModel.DEFAULT_REPO_ID
            is_default = (target == OmniVoiceBaseModel.DEFAULT_REPO_ID)
            if is_default:
                label = f"{COL_DIM}(optional)"
            else:
                value = ellipsize_path_for_menu(target)
                label = make_currently_string(value)
            return f"Custom model {label}"

        def make_speed_label(_) -> str:
            speed = state.project.get_model_setting('omnivoice_local', 'speed')
            return f"Speed {make_currently_string(speed, default=OmniVoiceBaseModel.DEFAULT_SPEED, num_decimals=1)}"

        def make_steps_label(_) -> str:
            steps = state.project.get_model_setting('omnivoice_local', 'num_step')
            return f"Steps {make_currently_string(steps, default=OmniVoiceBaseModel.DEFAULT_STEPS)}"

        def on_clear_model_target(s: State, __: MenuItem) -> None:
            s.project.set_model_setting('omnivoice_local', 'target', "")
            s.project.save()
            Tts.set_model_params_using_project(s.project)
            _ = ModelWorker.clear_models_if_running_blocking()
            print_feedback("Cleared, will use default model")


        def make_items(_: State) -> list[MenuItem]:
            items = []

            items.extend(
                VoiceMenuShared.make_voice_sample_items(
                    state,
                    TtsModelType.require_by_id("omnivoice_local"),
                    no_samples_label=make_voice_label,
                )
            )

            instruct_items = VoiceMenuShared.make_voice_instructions_item(
                state, 'omnivoice_local', validate_omnivoice=True)
            # Only the group's first item carries the heading: the renderer
            # reprints a superlabel before every item that has one.
            instruct_items[0].superlabel = VOICE_ADVANCED_SUPERLABEL
            items.extend(instruct_items)

            items.append(
                MenuItem(make_target_label, lambda _, __: ask_target(state))
            )
            if state.project.get_model_setting('omnivoice_local', 'target'):
                items.append(MenuItem("Clear custom model", on_clear_model_target))

            steps_item = MenuItem(make_steps_label, lambda _, __: ask_steps(state.project))
            items.append(steps_item)

            speed_item = MenuItem(make_speed_label, lambda _, __: ask_speed(state.project))
            items.append(speed_item)

            cfg_item = MenuUtil.make_number_item(
                state=state,
                target=SettingRef("omnivoice_local", "cfg"),
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
                VoiceMenuShared.make_seed_item(state, SettingRef("omnivoice_local", "seed"))
            )

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_target(state: State) -> None:
    project = state.project
    model_name = project.get_tts_model_type().value.ui["short_name"]
    prompt = f"Enter huggingface repo id or local directory path to {model_name} model"
    prompt += f"\n{COL_DIM}Eg, \"k2-fsa/OmniVoice\" or \"/path/to/local/checkpoint\""
    VoiceMenuShared.ask_target(
        project=project,
        prompt=prompt,
        current_target=project.get_model_setting('omnivoice_local', 'target'),
        callback=lambda _, target: apply_target(state, target)
    )


def apply_target(state: State, target: str) -> None:
    project = state.project
    previous_target = project.get_model_setting('omnivoice_local', 'target')

    def revert() -> None:
        project.set_model_setting('omnivoice_local', 'target', previous_target)
        _ = ModelWorker.clear_models_if_running_blocking()

    project.set_model_setting('omnivoice_local', 'target', target)
    _ = ModelWorker.clear_models_if_running_blocking()

    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        revert()
        ask.ask_error(f"Failed to load OmniVoice model: {error}")
        return

    project.save()
    print_feedback("Model set:", target)


def ask_speed(project: Project) -> None:
    ask.ask_number_and_save(
        project,
        SettingRef("omnivoice_local", "speed"),
        "Enter speech speed",
        0.5,
        2.0,
        OmniVoiceBaseModel.DEFAULT_SPEED,
        "Speed set:",
        is_minus_one_default=True,
    )


def ask_steps(project: Project) -> None:
    ask.ask_number_and_save(
        project,
        SettingRef("omnivoice_local", "num_step"),
        "Enter number of inference steps\nLarger values = enhanced quality, slower",
        OmniVoiceBaseModel.MIN_STEPS,
        OmniVoiceBaseModel.MAX_STEPS,
        OmniVoiceBaseModel.DEFAULT_STEPS,
        "Inference steps set:",
        is_int=True,
        is_minus_one_default=True,
    )
