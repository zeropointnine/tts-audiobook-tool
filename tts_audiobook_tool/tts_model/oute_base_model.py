from pathlib import Path

from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.util import ellipsize_path_for_menu
from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM, COL_ERROR

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class OuteBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.OUTE.value

    DEFAULT_TEMPERATURE = 0.4 # from model library code

    def create_speaker(self, path: str) -> dict | str:
        ...

    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[str, str]:
        """
        Override to use oute_voice_file_name (string path) instead of oute_voice_json (dict).
        """
        voice_file_name = getattr(project, "oute_voice_file_name", "")
        voice_file_name = ellipsize_path_for_menu(voice_file_name)

        match (cls.INFO.requires_voice, bool(voice_file_name)):
            case (True, False):
                prefix = COL_ERROR + "required"
                value = ""
            case (True, True):
                prefix = COL_DIM + "current voice clone"
                value = COL_ACCENT + voice_file_name
            case (False, False):
                prefix = COL_DIM + "current voice clone"
                value = COL_ERROR + "none"
            case (False, True):
                prefix = COL_DIM + "current voice clone"
                value = COL_ACCENT + voice_file_name

        return prefix, value

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:
        """
        Override to use oute_voice_file_name (string path) instead of oute_voice_json (dict).
        """
        voice_file_name = getattr(project, "oute_voice_file_name", "")
        if not voice_file_name:
            return "none"

        # Remove file suffix
        voice_file_name = Path(voice_file_name).stem

        # Remove filename 'postfix decorator'
        voice_file_name = voice_file_name.strip("_" + cls.INFO.file_tag)

        tag = TextUtil.sanitize_for_filename(voice_file_name[:30])
        return tag

    @classmethod
    def get_strictness_warning(cls, strictness: Strictness, project: Project, instance: TtsBaseModel | None) -> str:
        if strictness >= Strictness.HIGH:
            return "Not recommended with current TTS model"
        return ""
