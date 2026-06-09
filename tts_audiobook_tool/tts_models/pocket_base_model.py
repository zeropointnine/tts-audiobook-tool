from __future__ import annotations

import os

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class PocketBaseModel(TtsBaseModel):

    INFO = TtsModelType.POCKET.value

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_INT8 = False
    TEMPERATURE_MIN = 0.01
    TEMPERATURE_MAX = 2.0
    
    # This is pocket default value
    # Higher value may improve sound quality but I didn't hear it, 
    # so leaving it 'un-parameterized'
    LSD = 1 

    # High value here prevents pocket from doing its own internal chunking
    MAX_TOKENS = 120 

    # Predefined voices - stored as bare names, resolved by pocket_tts internals
    # (corresponds to https://huggingface.co/kyutai/pocket-tts/tree/main/embeddings_v3)
    PREDEFINED_VOICES = [
        "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles",
        "cosette", "eponine", "eve", "fantine", "george", "jane", "javert",
        "jean", "marius", "mary", "michael", "paul", "peter_yearsley",
        "stuart_bell", "vera",
    ]

    # See https://huggingface.co/kyutai/pocket-tts/tree/main/languages
    LANGUAGES = [
        "english_2026-04", # same as unqualified "english" 
        "english_2026-01", 
        "french_24l",
        "german_24l",
        "italian",
        "portuguese",
        "spanish_24l",
    ]
    DEFAULT_LANGUAGE = "english_2026-04"
    # Cache the gated/opt-in validation result per resolved voice prompt path so
    # repeated readiness checks (eg, menu label rendering) do not keep re-encoding
    # the same audio prompt through pocket_tts. This is intentionally process-
    # lifetime cache; changing voice path naturally uses a different key.
    gated_error_message_cache: dict[str, str] = {}

    @classmethod
    def get_voice_value(cls, project: Project) -> str:
        return project.pocket_voice_file_name or project.pocket_predefined_voice

    @classmethod
    def get_model_display_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        s = f"{cls.INFO.ui['proper_name']} {COL_DIM}{project.pocket_model_code}"
        return s

    @classmethod
    def get_blocking_issues(cls, project: Project, instance: TtsBaseModel | None) -> list[ReadinessIssue]:
        errors = []
        
        if instance:
            assert isinstance(instance, PocketBaseModel)
            if PocketBaseModel.get_gated_error_message(project, instance):
                verbose_ui_message = cls.make_gated_error_message_ui()
                errors.append(ReadinessIssue("ungated model", verbose_ui_message))
                return errors # don't bother adding any other errors at this point
        
        if not project.pocket_voice_file_name and not project.pocket_predefined_voice:
            errors.append(ReadinessIssue("voice clone", "Setting a voice clone file or predefined voice is required"))

        return errors

    # ---

    @classmethod
    def get_gated_error_message(cls, project: Project, instance: TtsBaseModel) -> str:
        """
        Returns error string if has voice clone and Pocket voice-clone-specific model is gated.
        Empty string means validation passed or no voice clone sample is set.

        Rem, use "make_gated_error_message_ui" for user-facing error message with remediation info
        """
        voice_file_name = project.pocket_voice_file_name
        if not voice_file_name:
            return ""

        assert isinstance(instance, PocketBaseModel)
        voice_path = os.path.join(project.dir_path, voice_file_name)
        cached = cls.gated_error_message_cache.get(voice_path, None)
        if cached is not None:
            return cached
        
        # We're avoiding referencing PocketModel directly here...
        validate_method = getattr(instance, "get_voice_clone_access_error_for_path", None)
        assert validate_method

        result = validate_method(voice_path)
        message = result if isinstance(result, str) else str(result)
        cls.gated_error_message_cache[voice_path] = message
        return message

    @classmethod
    def is_opt_in_error_string(cls, error: str) -> bool:
        error_lower = error.lower()
        return (
            "accept the terms" in error_lower
            or "voice cloning" in error_lower and "hf auth login" in error_lower
            or "model with voice cloning" in error_lower
        )

    @classmethod
    def make_gated_error_message_ui(cls) -> str:
        url = cls.INFO.ui["project_links"][1]
        return (
            "-----------------------------------------------------------------------------------\n"
            f"{COL_ERROR}{Ansi.ITALICS}Pocket voice cloning requires Hugging Face opt-in.{COL_DEFAULT}\n"
            "-----------------------------------------------------------------------------------\n"
            f"{OPT_IN_INSTRUCTIONS.replace('%1', url)}\n"
            "-----------------------------------------------------------------------------------"
        )
