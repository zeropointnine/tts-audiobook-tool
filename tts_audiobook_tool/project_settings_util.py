import json
from pathlib import Path

from tts_audiobook_tool.l import L
from tts_audiobook_tool.state import State

from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class ProjectSettingsUtil:
    """
    Settings for the current project.
    Currently stores just the project dir itself.
    """

    @staticmethod
    def load_from_disk(state: State) -> str:
        """
        Loads project dir path onto state object
        Returns user-facing error message on fail
        """

        if not os.path.exists(ProjectSettingsUtil.get_file_path()):
            return ""

        try:
            with open(ProjectSettingsUtil.get_file_path(), 'r', encoding='utf-8') as f:
                dic = json.load(f)
                if not isinstance(dic, dict):
                    return "Settings file error: Bad data"
        except Exception as e:
            return f"Settings file error: {e}"

        if not "project_dir_path" in dic:
            return "Settings file error: Missing data"

        dir_path = dic["project_dir_path"]
        if not os.path.exists(dir_path):
            return f"Project directory does not exist: {dir_path}"

        state.project_dir = dir_path
        return ""

    @staticmethod
    def save_to_disk(state: State) -> None:
        """
        Saves project dir path from state object
        Fails silently # TODO revisit?
        """
        dic = {
            "project_dir_path": state.project_dir
        }
        try:
            with open(ProjectSettingsUtil.get_file_path(), 'w', encoding='utf-8') as f:
                json.dump(dic, f, indent=4)
        except Exception as e:
            L.w(f"{e}")

    @staticmethod
    def get_file_path() -> str:
        dir = Path.home()
        return os.path.join(dir, SETTINGS_FILE_NAME)
