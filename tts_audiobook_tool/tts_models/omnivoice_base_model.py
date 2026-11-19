# tts_audiobook_tool/tts_model/omnivoice_base_model.py
from __future__ import annotations

from tts_audiobook_tool import util
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_models.language_resolution import LanguageResolution
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.local_definition import require_local_definition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class OmniVoiceBaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("omnivoice_local").value

    DEFAULT_REPO_ID = "k2-fsa/OmniVoice"
    # Speed, CFG and steps defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("omnivoice_local")

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        if not project.get_model_setting('omnivoice_local', 'target') or project.get_model_setting('omnivoice_local', 'target') == OmniVoiceBaseModel.DEFAULT_REPO_ID:
            s = cls.INFO.ui.get("proper_name") or ""
        else:
            s = cls.INFO.ui.get("short_name") or ""
            target = util.ellipsize_path_for_menu(project.get_model_setting('omnivoice_local', 'target'))
            s += f" {COL_DIM}({target})"
        return s

    @property
    def has_lora(self) -> bool:
        """Whether the LoRA set in the project was merged into this instance."""
        return False

    @property
    def lora_error(self) -> str:
        """Why the project's LoRA couldn't be loaded, or empty."""
        return ""

    @classmethod
    def get_blocking_issues(cls, project: Project, instance: TtsBaseModel | None) -> list[ReadinessIssue]:
        issues = super().get_blocking_issues(project, instance)
        if instance and project.get_model_setting('omnivoice_local', 'lora_target'):
            assert isinstance(instance, OmniVoiceBaseModel)
            if not instance.has_lora:
                issues.append(ReadinessIssue("valid LoRA", "Couldn't load LoRA"))
        return issues

    @classmethod
    def has_fixed_default_voice(cls, project: Project) -> bool:
        # A LoRA gives sample-less voices a fixed speaker (voice_variation_warning)
        return bool(project.get_model_setting('omnivoice_local', 'lora_target'))

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:
        # As for VibeVoice, a LoRA is the more salient identifier, even with a sample
        lora_target = project.get_model_setting('omnivoice_local', 'lora_target')
        if not lora_target:
            return super().get_voice_tag(project)
        from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
        from tts_audiobook_tool.target_util import make_target_display_name
        return ProjectVoiceUtil.make_voice_tag(make_target_display_name(lora_target))

    @classmethod
    def resolve_language(cls, project: Project, instance: TtsBaseModel | None = None) -> LanguageResolution:
        # An OmniVoice id, or none (language-agnostic mode) when empty or unknown
        from tts_audiobook_tool.tts_models.omnivoice_language import resolve_omnivoice_language
        return resolve_omnivoice_language(project.language_code)

    @classmethod
    def validate_instructions(cls, value: str) -> str:
        # Lazy: imports OmniVoice (not the model) on first use
        from tts_audiobook_tool.tts_models.omnivoice_instruct_util import validate_omnivoice_instruct
        return validate_omnivoice_instruct(value)

    def generate(
        self,
        text: str,
        speed: float,
        ref_audio_path: str = "",
        ref_text: str = "",
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
