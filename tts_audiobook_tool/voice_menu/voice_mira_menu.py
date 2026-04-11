from tts_audiobook_tool.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceMiraMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def on_clear_voice() -> None:
            if Tts.get_instance_if_exists():
                Tts.get_mira().clear_voice_clone()

        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.MIRA)
                )                
            )
            if state.project.mira_voice_file_name:
                items.append( VoiceMenuShared.make_clear_voice_item(
                    state, TtsModelInfos.MIRA, on_clear_voice
                ))

            item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="mira_temperature",
                default_value=MiraBaseModel.TEMPERATURE_DEFAULT,
                min_value=MiraBaseModel.TEMPERATURE_MIN,
                max_value=MiraBaseModel.TEMPERATURE_MAX
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            item = VoiceMenuShared.make_top_p_item(
                state=state,
                attr="mira_top_p",
                default_value=MiraBaseModel.TOP_P_DEFAULT
            )
            items.append(item)

            item = VoiceMenuShared.make_top_k_item(
                state=state,
                attr="mira_top_k",
                default_value=MiraBaseModel.TOP_K_DEFAULT
            )
            items.append(item)

            item = VoiceMenuShared.make_repetition_penalty_item(
                state=state,
                attr="mira_repetition_penalty",
                default_value=MiraBaseModel.REPETITION_PENALTY_DEFAULT
            )
            items.append(item)

            prompt = f"Enter a static seed value {COL_DIM}(or -1 for random){COL_DEFAULT}"
            prompt += f"\n{COL_DIM}(Note, audio generations are not idempotent when using batch mode): "
            items.append(VoiceMenuShared.make_seed_item(state, "mira_seed", prompt_override=prompt))
            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)
