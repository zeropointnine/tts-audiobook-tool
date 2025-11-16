from __future__ import annotations

from tts_audiobook_tool.app_types import RealTimeSubmenuState
from tts_audiobook_tool.ask_util import AskUtil
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
    Most importantly, `project` and `prefs`
    """

    _prefs: Prefs
    _project: Project


    def __init__(self):

        self.prefs = Prefs.load()

        self.real_time = RealTimeSubmenuState()

        if not self.prefs.project_dir:
            self.project = Project("")
        else:
            result = Project.load_using_dir_path(self.prefs.project_dir)
            if isinstance(result, str):
                AskUtil.ask_error(result)
                self.prefs.project_dir = ""
                self.project = Project("")
            else:
                self.project = result


    @property
    def project(self) -> Project:
        return self._project

    @project.setter
    def project(self, value: Project) -> None:
        self._project = value

        # Sync static values
        Tts.set_model_params_using_project(self.project)
        if not self.real_time.custom_text_groups:
            self.real_time.line_range = None


    @property
    def prefs(self) -> Prefs:
        return self._prefs

    @prefs.setter
    def prefs(self, value: Prefs) -> None:
        self._prefs = value

        # Sync static values
        Stt.set_variant(self.prefs.stt_variant)
        Stt.set_config(self.prefs.stt_config)

    def make_new_project(self, path: str) -> str:
        """
        Inits project directory and sets new project instance
        Return error string on fail
        """

        try:
            project_dir_path = Path(path).expanduser()
        except:
            return "Bad path"

        if not project_dir_path.is_absolute():
            if os.name == "posix":
                return "Please use either an absolute path or a relative path that starts with \"~\""
            else: # ie, Windows
                return "Please use absolute path"

        if project_dir_path.exists():
            # If exists, make sure dir is empty
            if os.listdir(project_dir_path):
                return "Directory is not empty"
        else:
            # Make project dir
            try:
                os.mkdir(project_dir_path) # note, not "make_dirs()"
            except Exception as e:
                return f"Error creating directory: {e}"

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
        self.prefs.project_dir = str(project_dir_path)
        self.project = Project( str(project_dir_path) )

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
            AskUtil.ask_error(result)
            self.project = Project("")
        else:
            self.project = result

    def reset(self):
        self.prefs.project_dir = ""
        self.project = Project("")
        self.real_time = RealTimeSubmenuState()

