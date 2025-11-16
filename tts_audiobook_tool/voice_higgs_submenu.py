from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_model import HiggsProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceHiggsSubmenu:

    @staticmethod
    def menu(state: State) -> None:

        project = state.project

        def make_temperature_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.higgs_temperature, HiggsProtocol.DEFAULT_TEMPERATURE, 2
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                "Enter temperature (0.01 to 2.0):",
                0.01, 2.0, # sane range IMO
                "higgs_temperature",
                "Temperature set to:"
            )

        items = [
            MenuItem(
                VoiceSubmenuShared.make_select_voice_label,
                lambda _, __: VoiceSubmenuShared.ask_and_set_voice_file(state, TtsModelInfos.HIGGS)
            ),
            VoiceSubmenuShared.make_clear_voice_item(state, TtsModelInfos.HIGGS),
            MenuItem(
                make_temperature_label,
                on_temperature
            )
        ]
        VoiceSubmenuShared.show_voice_menu(state, items)
