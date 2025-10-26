from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import VibeVoiceProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceVibeVoiceSubmenu:

    @staticmethod
    def menu(state: State) -> None:

        project = state.project

        def make_model_path_label(_) -> str:
            value = project.vibevoice_model_path if project.vibevoice_model_path else "none"
            return f"VibeVoice custom model path {make_currently_string(value)}"

        def make_cfg_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.vibevoice_cfg, VibeVoiceProtocol.DEFAULT_CFG, 1
            )
            return f"CFG scale {make_currently_string(value)}"

        def on_cfg(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter CFG (1.3 to 7.0):",
                1.3, 7.0, # Sane range IMO
                "vibevoice_cfg",
                "CFG set to:"
            )

        def make_steps_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.vibevoice_steps, VibeVoiceProtocol.DEFAULT_NUM_STEPS, 0
            )
            return f"Steps {make_currently_string(value)}"

        def on_steps(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter num steps (1-30):",
                1, 30, # Sane range IMO
                "vibevoice_cfg",
                "Num steps set to:",
                is_int=True
            )

        items = [
            MenuItem(
                VoiceSubmenuShared.make_select_voice_label,
                lambda _, __: VoiceSubmenuShared.ask_and_set_voice_file(state, TtsModelInfos.VIBEVOICE)
            ),
            VoiceSubmenuShared.make_clear_voice_item(state, TtsModelInfos.VIBEVOICE),
            MenuItem(
                make_model_path_label,
                lambda _, __: ask_model_path(state.project)
            ),
            MenuItem(
                make_cfg_label,
                on_cfg
            ),
            MenuItem(
                make_steps_label,
                on_steps
            ),
        ]
        VoiceSubmenuShared.show_voice_menu(state, items)

# ---

def ask_model_path(project: Project) -> None: # type: ignore
    s = "Select local directory containing VibeVoice model (Hugging Face model repository format):"
    dir_path = AskUtil.ask_dir_path(s, s)
    if not dir_path:
        return
    if dir_path == project.vibevoice_model_path:
        print_feedback("Already set")
        return
    if dir_path and not os.path.exists(dir_path):
        print_feedback("No such directory", color_code=COL_ERROR)
        return
    apply_model_path_and_validate(project, dir_path)

def apply_model_path_and_validate(project: Project, path: str) -> None: # type: ignore

    project.vibevoice_model_path = path
    project.save()

    Tts.set_model_params_using_project(project)

    if not path:
        # No need to validate
        print_feedback(f"Set to none; will use default model {VibeVoiceProtocol.DEFAULT_MODEL_NAME}")
        return

    # Validate by attempting to instantiate model with new settings

    # Model should have been cleared, but just in case:
    model = Tts.get_instance_if_exists()
    if model:
        Tts.clear_tts_model()

    try:
        _ = Tts.get_vibevoice()
        print_feedback(f"\nCustom model path set: {path}")

    except (OSError, Exception) as e:
        # Revert change
        project.vibevoice_model_path = ""
        project.save()
        Tts.set_model_params_using_project(project)
        printt()
        printt(f"{COL_ERROR}Contents at model path {path} appear to be invalid:")
        printt(f"{COL_ERROR}{make_error_string(e)}")
        printt()
        AskUtil.ask_enter_to_continue()
