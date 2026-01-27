from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_model import FishProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceFishMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    VoiceMenuShared.make_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.FISH)
                )
            )
            if state.project.fish_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.FISH) 
                )

            items.append( 
                VoiceMenuShared.make_temperature_item(
                    state=state,
                    attr="fish_temperature",
                    default_value=FishProtocol.DEFAULT_TEMPERATURE,
                    min_value=0.01,
                    max_value=2.0
                )
            )

            items.append(
                VoiceMenuShared.make_seed_item(state, "fish_seed")
            )
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)
