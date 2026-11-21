from __future__ import annotations


from tts_audiobook_tool.constants import *
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.local_definition import require_local_definition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from typing import TYPE_CHECKING
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class PocketBaseModel(TtsBaseModel):

    INFO = TtsModelType.require_by_id("pocket_local").value

    # Parameter defaults and bounds (`[models.local.parameters]`); see `LOCAL.resolve()`
    LOCAL = require_local_definition("pocket_local")
    DEFAULT_INT8 = False

    # Pocket's `sampler_decode_steps` (formerly `lsd_decode_steps`); this is the
    # pocket default value. Higher value may improve sound quality but I couldn't
    # hear it, so leaving it 'un-parameterized'
    SAMPLER_DECODE_STEPS = 1

    # High value here prevents pocket from doing its own internal chunking
    MAX_TOKENS = 120

    # Predefined voices - stored as bare names, resolved by pocket_tts internals.
    # Every language model has an embedding for each of these
    # (https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/tree/main/languages).
    # The English set comes first, then the per-language defaults (it, es, de, pt, fr, nl).
    PREDEFINED_VOICES = [
        "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles",
        "cosette", "eponine", "eve", "fantine", "george", "jane", "javert",
        "jean", "marius", "mary", "michael", "paul", "peter_yearsley",
        "stuart_bell", "vera",
        "giovanni", "lola", "juergen", "rafael", "estelle", "daan",
    ]

    # Model codes (pocket_tts `config/<code>.yaml`): current models first (grouped
    # by language, recommended model first), superseded English versions last. Superseded codes are kept for backcompat: projects
    # store the code, and an in-progress book should be able to stay on its model.
    # Bare aliases ("english") are omitted because their meaning changes across releases.
    # See https://huggingface.co/kyutai/pocket-tts/tree/main/languages
    LANGUAGES = [
        # Current English, one entry (plus reward-tuned variant) per size
        "english_2026-10",
        "english_18m_2026-10",
        "english_18m_refl_2026-10",
        "english_7m_2026-10",
        "english_7m_refl_2026-10",
        "english_2m_2026-10",
        "english_2m_refl_2026-10",
        # Non-English, full 24-layer model before its distilled 6-layer one
        "french_24l",
        "french",
        "german_24l",
        "german",
        "spanish_24l",
        "spanish",
        "italian_24l",
        "italian",
        "portuguese_24l",
        "portuguese",
        "dutch_24l",
        "dutch",
        # Superseded/experimental English, kept selectable for backcompat. The English
        # 24-layer models are older teachers lacking later fixes (upstream only calls
        # the non-English 24-layer models higher quality).
        "english_2026-09_2",
        "english_2026-09",
        "english_2026-09_24l",
        "english_2026-04",
        "english_2026-04_24l",
        "english_2026-01",
        "english_drifting_26-09",
    ]
    # Short notes appended to model picker labels (values stay the bare codes)
    LANGUAGE_NOTES: dict[str, str] = {
        "english_2026-09_24l": "24 layers, older version",
        "english_2026-04_24l": "24 layers, older version",
        "english_2026-09_2": "older version",
        "english_2026-09": "older version",
        "english_2026-04": "older version",
        "english_2026-01": "older version",
        "english_drifting_26-09": "experimental sampler",
        # Param counts per upstream README (small models: 18.2M, 6.9M, 1.9M incl. decoder)
        "english_2026-10": "100M params",
        "english_18m_2026-10": "18M params",
        "english_18m_refl_2026-10": "18M params, reward-tuned",
        "english_7m_2026-10": "7M params",
        "english_7m_refl_2026-10": "7M params, reward-tuned",
        "english_2m_2026-10": "2M params, experimental",
        "english_2m_refl_2026-10": "2M params, reward-tuned, experimental",
        "french_24l": "24 layers, full model, higher quality, slower",
        "french": "6 layers, distilled, faster",
        "german_24l": "24 layers, full model, higher quality, slower",
        "german": "6 layers, distilled, faster",
        "spanish_24l": "24 layers, full model, higher quality, slower",
        "spanish": "6 layers, distilled, faster",
        "italian_24l": "24 layers, full model, higher quality, slower",
        "italian": "6 layers, distilled, faster",
        "portuguese_24l": "24 layers, full model, higher quality, slower",
        "portuguese": "6 layers, distilled, faster",
        "dutch_24l": "24 layers, full model, higher quality, slower",
        "dutch": "6 layers, distilled, faster",
    }

    # Model to suggest (picker label hint only) for a normalized project language code.
    # Non-English: the full 24-layer models, which upstream describes as higher
    # quality than the distilled 6-layer ones (audiobooks favor quality over speed).
    # English: english_2026-10, the newest, has no 24-layer variant; the English
    # 24-layer models are older versions.
    # There is deliberately no default model: a project must select one explicitly
    # (see get_blocking_issues), so a package update or new upstream models can't
    # silently change the model an in-progress book uses.
    RECOMMENDED_BY_LANGUAGE: dict[str, str] = {
        "en": "english_2026-10",
        "fr": "french_24l",
        "de": "german_24l",
        "es": "spanish_24l",
        "it": "italian_24l",
        "pt": "portuguese_24l",
        "nl": "dutch_24l",
    }

    MODEL_NOT_SELECTED_MESSAGE = (
        "No Pocket TTS model selected. Select one in Model settings > Pocket model"
    )

    @classmethod
    def get_recommended_language(cls, project: Project) -> str:
        """The suggested model for the project's language, or "" if Pocket has none."""
        from tts_audiobook_tool.text_ops.language_util import normalize_language_code
        return cls.RECOMMENDED_BY_LANGUAGE.get(normalize_language_code(project.language_code), "")

    @classmethod
    def make_language_label(cls, code: str, recommended: str = "") -> str:
        notes = [note for note in (
            "recommended" if code == recommended else "",
            cls.LANGUAGE_NOTES.get(code, ""),
        ) if note]
        return f"{code} {COL_DIM}({', '.join(notes)})" if notes else code
    # Cache the gated/opt-in validation result per resolved voice prompt path so
    # repeated readiness checks (eg, menu label rendering) do not keep re-encoding
    # the same audio prompt through pocket_tts. This is intentionally process-
    # lifetime cache; changing voice path naturally uses a different key.
    gated_error_message_cache: dict[str, str] = {}

    @classmethod
    def can_voice_presets(cls, project: Project, instance: TtsBaseModel | None = None) -> bool:
        return True

    @classmethod
    def get_voice_tag_for_preset(cls, preset: str) -> str:
        return ProjectVoiceUtil.make_voice_file_name_tag(preset, cls.INFO.file_tag)

    @staticmethod
    def get_primary_voice_preset(project: Project) -> str:
        voices = project.voice_references
        return voices[0].preset_for("pocket_local") if voices else ""

    @classmethod
    def get_primary_voice_value(cls, project: Project) -> str:
        return cls.get_primary_voice_preset(project) or ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.require_by_id("pocket_local"))

    @staticmethod
    def get_first_cloned_voice_file_name(project: Project) -> str:
        """The sample of the first voice that clones (has a sample and no
        predefined voice, which would take precedence), or ""."""
        for voice in project.voice_references:
            if voice.has_sample and not voice.preset_for("pocket_local"):
                return voice.file_name
        return ""

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        model_code = project.get_model_setting('pocket_local', 'model_code') or "model not selected"
        return f"{cls.INFO.ui['proper_name']} {COL_DIM}({model_code})"

    @classmethod
    def get_blocking_issues(cls, project: Project, instance: TtsBaseModel | None) -> list[ReadinessIssue]:
        errors = []

        # Checked without an instance, so menus and run starts block before any load
        if not project.get_model_setting('pocket_local', 'model_code'):
            errors.append(ReadinessIssue("Pocket model selection", cls.MODEL_NOT_SELECTED_MESSAGE))
            return errors

        if instance:
            assert isinstance(instance, PocketBaseModel)
            if PocketBaseModel.get_gated_error_message(project, instance):
                verbose_ui_message = cls.make_gated_error_message_ui()
                errors.append(ReadinessIssue("ungated model", verbose_ui_message))
                return errors # don't bother adding any other errors at this point

        return errors

    # ---

    @classmethod
    def get_gated_error_message(cls, project: Project, instance: TtsBaseModel) -> str:
        """
        Returns error string if has voice clone and Pocket voice-clone-specific model is gated.
        Empty string means validation passed or no voice clone sample is set.

        Rem, use "make_gated_error_message_ui" for user-facing error message with remediation info
        """
        voice_file_name = cls.get_first_cloned_voice_file_name(project)
        if not voice_file_name:
            return ""

        assert isinstance(instance, PocketBaseModel)
        voice_path = ProjectVoiceUtil.resolve_model_voice_file_path(project, voice_file_name)
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
