from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_model import FishProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu_shared import VoiceMenuShared

class VoiceFishMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        project = state.project

        def make_temperature_label(_) -> str:
            value = make_parameter_value_string(
                project.fish_temperature, FishProtocol.DEFAULT_TEMPERATURE, 1
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                "Enter temperature (0.01 to 2.0):",
                0.01, 2.0, # sane range IMO
                "fish_temperature",
                "Temperature set to:"
            )

        def make_items(_: State) -> list[MenuItem]:
            items = [
                MenuItem(
                    VoiceMenuShared.make_select_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.FISH)
                )
            ]
            if state.project.fish_voice_file_name:
                items.append( VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.FISH) )
            items.append( MenuItem(make_temperature_label, on_temperature) )
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)
