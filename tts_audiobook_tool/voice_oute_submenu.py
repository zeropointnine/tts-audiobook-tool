from pathlib import Path

from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model import OuteProtocol
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceOuteSubmenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        project = state.project

        def make_temperature_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                state.project.oute_temperature, OuteProtocol.DEFAULT_TEMPERATURE, 1
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter temperature (0.01 to 2.0):",
                0.01, 2.0,
                "oute_temperature",
                "Temperature set to:"
            )

        def on_default(_, __) -> None:
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                AskUtil.ask_error(result)
                return
            project.set_oute_voice_and_save(result, "default")
            print_feedback("Voice clone set.")

        items = [
            MenuItem(
                "Set voice clone using audio clip (up to 15s)",
                lambda _, __: ask_create_oute_voice(project)
            ),
            MenuItem(
                "Set voice clone using Oute json file",
                lambda _, __: ask_load_oute_json(project)
            ),
            MenuItem(
                "Clear voice clone (use Oute default)",
                on_default
            ),
            MenuItem(
                make_temperature_label,
                on_temperature
            )
        ]
        VoiceSubmenuShared.show_voice_menu(state, items)

# ---

def ask_create_oute_voice(project: Project) -> None:

    from tts_audiobook_tool.app_util import AppUtil

    path = VoiceSubmenuShared.ask_voice_file(project.dir_path, Tts.get_type())
    if not path:
        return

    # Outte is about to create its own instance of whisper, so better clear ours first
    Stt.clear_stt_model()
    AppUtil.gc_ram_vram()

    result = Tts.get_oute().create_speaker(path)

    # Clear lingering oute-created whisper instance
    AppUtil.gc_ram_vram()

    if isinstance(result, str):
        error = result
        AskUtil.ask_error(f"Error creating voice: {error}")
        return

    voice_dict = result
    project.set_oute_voice_and_save(voice_dict, Path(path).stem)

    printt()
    print_feedback("Voice clone set")


def ask_load_oute_json(project: Project):

    path = AskUtil.ask_file_path(
        "Enter file path of voice json file: ",
        "Select Oute voice json file",
        [("JSON files", "*.json"), ("All files", "*.*")],
        initialdir=project.dir_path
    )
    if not path:
        return

    result = OuteUtil.load_oute_voice_json(path)
    if isinstance(result, str):
        AskUtil.ask_error(result)
        return

    project.set_oute_voice_and_save(result, Path(path).stem)
    print_feedback("Voice clone set.")
