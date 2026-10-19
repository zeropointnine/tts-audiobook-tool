from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DIM_ITALICS, print_feedback, printt
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared


class VoicePocketMenu:

    @staticmethod
    def menu(state: State) -> None:
        def make_items(_: State) -> list[MenuItem]:
            return VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("pocket_local"),
                on_set_callback=lambda: validate_voice_file(state),
                preset_choices=get_predefined_voice_choices,
            )

        VoiceMenuShared.menu_wrapper(state, make_items)


def get_predefined_voice_choices(_: State) -> list[str]:
    return list(PocketBaseModel.PREDEFINED_VOICES)


def validate_voice_file(state: State) -> None:

    if state.pocket_voice_clone_access_validated:
        return

    # Only a voice that clones (no predefined voice taking precedence) needs access
    if PocketBaseModel.get_first_cloned_voice_file_name(state.project):

        printt(f"{COL_DIM_ITALICS}Validating Pocket voice cloning access...")
        printt()

        inspection, inspect_error = ModelWorker.inspect_tts_blocking(state)
        error = inspect_error
        if inspection is not None and inspection.blocking_issues:
            error = inspection.blocking_issues[0]
        if error:
            message = error
            if PocketBaseModel.is_opt_in_error_string(error):
                message = PocketBaseModel.make_gated_error_message_ui()
                # TMI: message += f"\n\n{COL_DIM}Underlying Pocket error:{COL_DEFAULT}\n{error}"
            message += "\n"
            ask.ask_error(message)
        elif inspection is not None:
            state.pocket_voice_clone_access_validated = True
            print_feedback("Validated")
