from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.util import COL_DIM, COL_DIM_ITALICS, COL_ERROR, ellipsize_path_for_menu, make_currently_string, print_feedback, printt
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared


class VoicePocketMenu:

    @staticmethod
    def menu(state: State) -> None:
        def make_voice_file_label(_: State) -> str:
            label = "Select voice clone sample"
            if not state.project.get_model_setting('pocket_local', 'file_name') and not state.project.get_model_setting('pocket_local', 'predefined_voice'):
                label += f" {COL_DIM}({COL_ERROR}voice clone or predefined voice required{COL_DIM})"
            else:
                value = ProjectVoiceUtil.get_primary_voice_value(state.project, TtsModelType.require_by_id("pocket_local"))
                if value:
                    currently = make_currently_string(ellipsize_path_for_menu(value.removesuffix("_pocket.flac")))
                    label += currently
            return label

        def make_predefined_voice_label(_: State) -> str:
            name = state.project.get_model_setting('pocket_local', 'predefined_voice')
            currently = make_currently_string(name) if name else ""
            label = f"Select predefined voice {currently}".strip()
            if name and state.project.voice_references:
                label += f" {COL_DIM}(supercedes voice clone sample)"
            return label

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("pocket_local"),
                no_samples_label=make_voice_file_label,
                on_set_callback=lambda: validate_voice_file(state),
            )
            items.append(
                MenuItem(
                    make_predefined_voice_label,
                    lambda _, __: select_predefined_voice(state),
                    blank_line_before=True,
                )
            )
            if state.project.get_model_setting('pocket_local', 'predefined_voice'):
                items.append(
                    MenuItem(
                        "Clear predefined voice",
                        lambda _, __: clear_predefined_voice(state),
                    )
                )
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


def select_predefined_voice(state: State) -> None:
    voices = PocketBaseModel.PREDEFINED_VOICES

    current = state.project.get_model_setting('pocket_local', 'predefined_voice') or None

    def on_select(voice: str) -> None:
        state.project.set_model_setting('pocket_local', 'predefined_voice', voice)
        state.project.save()

    MenuUtil.options_menu(
        state=state,
        heading_text="Select predefined voice",
        labels=voices,
        values=voices,
        current_value=current,
        default_value=None,
        on_select=on_select,
    )


def clear_predefined_voice(state: State) -> None:
    state.project.set_model_setting('pocket_local', 'predefined_voice', "")
    state.project.save()
    print_feedback("Cleared")

def validate_voice_file(state: State) -> None:

    if state.pocket_voice_clone_access_validated:
        return

    if state.project.get_model_setting('pocket_local', 'file_name') and not state.project.get_model_setting('pocket_local', 'predefined_voice'):

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
