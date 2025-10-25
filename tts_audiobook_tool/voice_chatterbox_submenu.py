from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_model import ChatterboxProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceChatterboxSubmenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        project = state.project

        def on_clear_voice(_, __) -> None:
            project.clear_voice_and_save(TtsModelInfos.CHATTERBOX)
            printt_set("Cleared")

        def make_temperature_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.chatterbox_temperature, ChatterboxProtocol.DEFAULT_TEMPERATURE, 1
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter temperature (0.01 to 2.0):",
                0.01, 2.0,
                "chatterbox_temperature",
                "Temperature set to:"
            )

        def make_exagg_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.chatterbox_exaggeration, ChatterboxProtocol.DEFAULT_EXAGGERATION, 2
            )
            return f"Exaggeration {make_currently_string(value)}"

        def on_exagg(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter value for exaggeration (0.25 to 2.0):",
                0.25, 2.0,
                "chatterbox_exaggeration",
                "Exaggeration set to:"
            )

        def make_cfg_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.chatterbox_cfg, ChatterboxProtocol.DEFAULT_CFG, 2
            )
            return f"CFG/pace {make_currently_string(value)}"

        def on_cfg(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter value for exaggeration (0.2 to 1.0):",
                0.25, 1.0,
                "chatterbox_cfg",
                "Exaggeration set to:"
            )

        items = [
            MenuItem(
                VoiceSubmenuShared.make_select_voice_label,
                lambda _, __: VoiceSubmenuShared.ask_and_set_voice_file(state, TtsModelInfos.CHATTERBOX)
            ),
            MenuItem(
                "Clear voice clone sample",
                on_clear_voice
            ),
            MenuItem(
                make_temperature_label,
                on_temperature
            ),
            MenuItem(
                make_exagg_label,
                on_exagg
            ),
            MenuItem(
                make_cfg_label,
                on_cfg
            )
        ]
        MenuUtil.menu(state, VoiceSubmenuShared.MENU_HEADING, items)
