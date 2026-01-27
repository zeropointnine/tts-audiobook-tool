from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import MiraProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceMiraMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        project = state.project

        def make_temperature_label(_) -> str:
            value = make_parameter_value_string(
                project.mira_temperature, MiraProtocol.TEMPERATURE_DEFAULT, 1
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                f"Enter temperature ({MiraProtocol.TEMPERATURE_MIN} to {MiraProtocol.TEMPERATURE_MAX}):",
                MiraProtocol.TEMPERATURE_MIN, MiraProtocol.TEMPERATURE_MAX, 
                "mira_temperature",
                "Temperature set to:"
            )

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
            items.append( MenuItem(make_temperature_label, on_temperature) )
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)
