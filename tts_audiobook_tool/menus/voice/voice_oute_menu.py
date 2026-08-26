from pathlib import Path

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.tts_models.oute_util import OuteUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.oute_base_model import OuteBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceOuteMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        project = state.project

        def make_temperature_label(_) -> str:
            value = make_parameter_value_string(
                state.project.oute_temperature, OuteBaseModel.DEFAULT_TEMPERATURE, 1
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_: State, __: MenuItem) -> None:
            ask.ask_number_and_save(
                project,
                "oute_temperature",
                "Enter temperature:",
                0.01, 2.0,
                OuteBaseModel.DEFAULT_TEMPERATURE,
                "Temperature set to:"
            )

        def on_default(_: State, __: MenuItem) -> None:
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                ask.ask_error(result)
                return
            project.set_oute_voice_and_save(result, "default")
            print_feedback("Voice clone set")

        items = [
            MenuItem(
                "Set voice clone using audio clip (up to 15s)",
                lambda _, __: ask_create_oute_voice(state)
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
        VoiceMenuShared.menu_wrapper(state, items)

# ---

def ask_create_oute_voice(state: State) -> None:
    project = state.project

    path = VoiceMenuShared.ask_voice_file(project.dir_path, Tts.get_type())
    if not path:
        return

    result, error = ModelWorker.create_oute_speaker_blocking(state, path)

    if error or result is None:
        ask.ask_error(f"Error creating voice: {error}")
        return

    project.set_oute_voice_and_save(result, Path(path).stem)

    printt()
    print_feedback("Voice clone set")


def ask_load_oute_json(project: Project):

    path = ask.ask_file_path(
        "Enter file path of voice json file: ",
        "Select Oute voice json file",
        [("JSON files", "*.json"), ("All files", "*.*")],
        initialdir=project.dir_path
    )
    if not path:
        return

    result = OuteUtil.load_oute_voice_json(path)
    if isinstance(result, str):
        ask.ask_error(result)
        return

    project.set_oute_voice_and_save(result, Path(path).stem)
    print_feedback("Voice clone set")
