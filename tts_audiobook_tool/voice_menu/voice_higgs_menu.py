from tts_audiobook_tool.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_model.higgs_base_model import HiggsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceHiggsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            items = [
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.HIGGS)
                )
            ]
            if state.project.higgs_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.HIGGS) 
                )
            items.append( 
                VoiceMenuShared.make_temperature_item(
                    state=state,
                    attr="higgs_temperature",
                    default_value=HiggsBaseModel.DEFAULT_TEMPERATURE,
                    min_value=0.01,
                    max_value=2.0
                )
            )
            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)
