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
                    VoiceMenuShared.make_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.MIRA)
                )                
            )
            if state.project.mira_voice_file_name:
                items.append( VoiceMenuShared.make_clear_voice_item(
                    state, TtsModelInfos.MIRA, on_clear_voice
                ))
            items.append(
                VoiceMenuShared.make_temperature_item(
                    state=state,
                    attr="mira_temperature",
                    default_value=MiraBaseModel.TEMPERATURE_DEFAULT,
                    min_value=MiraBaseModel.TEMPERATURE_MIN,
                    max_value=MiraBaseModel.TEMPERATURE_MAX
                )
            )
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)
