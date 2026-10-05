from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_ERROR, make_currently_string
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared


class VoiceOmniVoiceMenu:

    @staticmethod
    def menu(state: State) -> None:
        def make_voice_label(_) -> str:
            if not state.project.get_model_setting('omnivoice_local', 'file_name'):
                currently = make_currently_string("none", value_prefix="", color_code=COL_ERROR)
            else:
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
            return f"Select voice clone sample {currently}"

        def make_items(_: State) -> list[MenuItem]:
            from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

            items = VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("omnivoice_local"),
                no_samples_label=make_voice_label,
            )
            instruct_items = ModelMenuShared.make_voice_instructions_item(
                state, 'omnivoice_local', validate_omnivoice=True)
            instruct_items[0].blank_line_before = True
            items.extend(instruct_items)
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
