# tts_audiobook_tool/tts_model/omnivoice_base_model.py
from __future__ import annotations

from tts_audiobook_tool import util
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class OmniVoiceBaseModel(TtsBaseModel):

    INFO = TtsModelType.OMNIVOICE.value

    DEFAULT_REPO_ID = "k2-fsa/OmniVoice"
    SAMPLE_RATE     = 24_000   # OmniVoice's default generation frequency
    DEFAULT_SPEED   = 1.0
    CFG_DEFAULT     = 2.0
    CFG_MIN         = 0.0
    CFG_MAX         = 4.0
    DEFAULT_STEPS   = 32
    MIN_STEPS       = 8
    MAX_STEPS       = 64

    @classmethod 
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        if not project.omnivoice_target or project.omnivoice_target == OmniVoiceBaseModel.DEFAULT_REPO_ID:
            s = cls.INFO.ui.get("proper_name") or ""
        else:
            s = cls.INFO.ui.get("short_name") or ""
            target = util.ellipsize_path_for_menu(project.omnivoice_target)
            s += f" {COL_DIM}({target})"
        return s

    def generate(
        self,
        text: str,
        ref_audio_path: str = "",
        ref_text: str = "",
        speed: float = DEFAULT_SPEED,
        instruct: str = "",
        language_id: str = "",
        duration: float | None = None,
        **kwargs,
    ) -> tuple[list, int]:
        """
        Returns ([audio_array_float32_mono], sample_rate).
        """
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError
