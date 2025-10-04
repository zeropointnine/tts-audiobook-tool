from __future__ import annotations

from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class State:
    """
    Holds app's user-related state
    Viz: current project and prefs
    """

    def __init__(self):
        self.prefs = Prefs.load()

        if not self.prefs.project_dir:
            self.project = Project("")
        else:
            result = Project.load_using_dir_path(self.prefs.project_dir)
            if isinstance(result, str):
                ask_error(result)
                self.prefs.project_dir = ""
                self.project = Project("")
            else:
                self.project = result

    @property
    def project(self) -> Project:
        return self._project

    @project.setter
    def project(self, project: Project) -> None:
        self._project = project

        # Sync global values :/
        Tts.set_model_params_using_project(self.project)
        Stt.set_variant_using_project(self.project)


    def make_new_project(self, path: str) -> str:
        """
        Inits project directory and sets new project instance
        Return error string on fail
        """

        try:
            project_dir_path = Path(path)
        except:
            return "Bad path"

        if project_dir_path.exists():
            # If exists, make sure dir is empty
            if os.listdir(project_dir_path):
                return "Directory is not empty"
        else:
            # Make project dir
            try:
                os.makedirs(project_dir_path, exist_ok=True)
            except:
                return "Error creating directory"

        # Make subdirs
        try:
            # Make audio segments subdir
            audio_segments_path = project_dir_path / PROJECT_SOUND_SEGMENTS_SUBDIR
            os.makedirs(audio_segments_path, exist_ok=True)
            # Make concat subdir
            concat_path = project_dir_path / PROJECT_CONCAT_SUBDIR
            os.makedirs(concat_path, exist_ok=True)
        except Exception as e:
            return f"Error creating subdirectory"

        # Make project instance
        self.prefs.project_dir = str( project_dir_path )
        self.project = Project(path)

        if Tts.get_type() == TtsModelInfos.OUTE:
            # Set Oute default voice
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                printt(result) # not ideal
            else:
                self.project.set_oute_voice_and_save(result, "default")

        self.project.save()

        return ""


    def set_existing_project(self, path: str) -> None:
        self.prefs.project_dir = path
        result = Project.load_using_dir_path(path)
        if isinstance(result, str):
            ask_error(result)
            self.project = Project("")
        else:
            self.project = result

    def reset(self):
        self.prefs = Prefs.new_and_save()
        self.project = Project("")
