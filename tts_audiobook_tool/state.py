from __future__ import annotations

from dataclasses import dataclass

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool import ask
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.text_ops.whitelist import Whitelist

@dataclass(frozen=True)
class PendingTtsModelChange:
    old_model_id: str
    new_model_id: str


class State:
    """
    Holds:
    - Persistent app state (via Prefs)
    - Project state
    - Some transitory app state (minor)
    """

    _prefs: Prefs
    _project: Project
    pending_tts_model_change: PendingTtsModelChange | None
    pending_project_load_checks: bool


    def __init__(self):

        # The initial sound-segment scan happens while the first main menu is
        # being assembled. Suppress its transient progress line until that
        # menu has been drawn once.
        self.dont_show_scan_message = True
        self.has_shown_main_menu = False
        # Remember successful Pocket clone access for this app run only.
        self.pocket_voice_clone_access_validated = False
        self.pending_tts_model_change = None
        self.pending_project_load_checks = False

        self.prefs = Prefs.load()

        self._project = None # type: ignore

        if not self.prefs.project_dir:
            self.project = Project(dir_path="")
        else:
            result = ProjectLoadUtil.load_using_dir_path(self.prefs.project_dir)
            if isinstance(result, str):
                ask.ask_error(result)
                self.prefs.project_dir = ""
                self.prefs.save()
                self.project = Project(dir_path="")
            else:
                self.project = result

    @classmethod
    def for_worker(cls, prefs: Prefs) -> "State":
        """
        Process-local State for the spawned model worker.

        Mirrors exactly the instance attributes set by `__init__`, but skips
        the interactive project load from mutable global prefs (the worker
        loads its own Project and assigns it through the normal setter).
        If `__init__` gains a new instance attribute, mirror it here.
        """
        state = cls.__new__(cls)
        state._project = None  # type: ignore[assignment]
        state.dont_show_scan_message = False
        state.has_shown_main_menu = False
        state.pocket_voice_clone_access_validated = False
        state.pending_tts_model_change = None
        state.pending_project_load_checks = False
        state.prefs = prefs
        return state

    @property
    def project(self) -> Project:
        return self._project

    @project.setter
    def project(self, value: Project) -> None:
        from tts_audiobook_tool.tts import Tts

        if self._project and self._project != value:
            self._project.kill()

        self.pending_tts_model_change = None
        self.pending_project_load_checks = bool(value.dir_path)
        self._project = value
        self._project.sound_segments.dont_show_scan_message = self.dont_show_scan_message

        # Sync static values
        Tts.bind_project(self.project)
        Whitelist().set_language_code(self.project.language_code)

    def mark_main_menu_shown(self) -> None:
        if self.has_shown_main_menu:
            return
        self.has_shown_main_menu = True
        self.dont_show_scan_message = False
        self.project.sound_segments.dont_show_scan_message = False

    @property
    def prefs(self) -> Prefs:
        return self._prefs

    @prefs.setter
    def prefs(self, value: Prefs) -> None:
        from tts_audiobook_tool.tts import Tts
        self._prefs = value
        # Sync static values
        Stt.set_variant(self.prefs.stt_variant)
        Stt.set_config(self.prefs.stt_config)
        Tts.set_force_cpu(self.prefs.tts_force_cpu)
        from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
        RemoteTtsDiscovery.set_base_url(self.prefs.remote_tts_url)
        SglOmniUtil.set_base_url(self.prefs.remote_tts_url)
        from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
        AudioCppUtil.set_base_url(self.prefs.remote_tts_url)

    @staticmethod
    def prepare_new_project_directory(path: str) -> str:
        """Create an empty project directory without selecting or saving a project."""
        try:
            project_dir_path = Path(path).expanduser()
        except:
            return "Bad path"
        if not project_dir_path.is_absolute():
            return "Please use an absolute path"

        if project_dir_path.exists() and not project_dir_path.is_dir():
            return f"Not a directory: {project_dir_path}"

        if project_dir_path.exists() and os.listdir(project_dir_path):
            return "Directory is not empty"

        # Make project dir
        if not project_dir_path.exists():
            try:
                os.mkdir(project_dir_path) # note, not using `make_dirs()``
            except Exception as e:
                return f"Error creating directory: {e}"

        # Make subdirs
        try:
            # Make sound segments subdir
            audio_segments_path = project_dir_path / PROJECT_SOUND_SEGMENTS_SUBDIR
            os.makedirs(audio_segments_path, exist_ok=True)
            # Make concat subdir
            concat_path = project_dir_path / PROJECT_CONCAT_SUBDIR
            os.makedirs(concat_path, exist_ok=True)
        except Exception as e:
            return make_error_string(e)

        return ""

    def make_and_set_new_project(self, path: str) -> str:
        """Initialize and select a new project; return an error string on failure."""
        err = self.prepare_new_project_directory(path)
        if err:
            return err
        project_dir_path = Path(path).expanduser()

        # Make project
        self.prefs.project_dir = str(project_dir_path)
        self.prefs.save()
        from tts_audiobook_tool.tts import Tts
        available = Tts.get_available_tts_models()
        model_id = available[0].id if len(available) == 1 else "none"
        self.project = Project(dir_path=str(project_dir_path), tts_model_type=model_id)

        self.project.save()
        return ""

    def set_existing_project(self, path: str) -> None:
        self.prefs.project_dir = path
        self.prefs.save()
        result = ProjectLoadUtil.load_using_dir_path(path)
        if isinstance(result, str):
            ask.ask_error(result)
            self.project = Project(dir_path="")
        else:
            self.project = result

    def reset(self):
        self.prefs.project_dir = ""
        self.prefs.save()
        self.project = Project(dir_path="")
