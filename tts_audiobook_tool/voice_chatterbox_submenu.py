from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_model import ChatterboxProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared
from tts_audiobook_tool.ask_util import AskUtil

class VoiceChatterboxSubmenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        project = state.project

        def make_language_label(_) -> str:
            # Safely get the language, default to 'de' if missing
            lang = getattr(project, "chatterbox_language", ChatterboxProtocol.DEFAULT_LANGUAGE)
            if lang == ChatterboxProtocol.DEFAULT_LANGUAGE:
                 val_str = f"{lang} (default)"
            else:
                 val_str = lang
                 
            return f"Language Code {make_currently_string(val_str)}"

        def on_language(_, __) -> None:
            default_code = ChatterboxProtocol.DEFAULT_LANGUAGE
             
            VoiceSubmenuShared.ask_string(
                project,
                f"Enter language code (default '{default_code}'):",
                "chatterbox_language",
                "Language set to:"
            )

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
            VoiceSubmenuShared.make_clear_voice_item(state, TtsModelInfos.CHATTERBOX),
            MenuItem(
                make_language_label,
                on_language
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
        VoiceSubmenuShared.show_voice_menu(state, items)
