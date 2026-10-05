from __future__ import annotations
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationInfo, field_validator, model_validator

from tts_audiobook_tool.app_support import path_norm
from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Book, BookSection, SectionMarkerMode, ExportType, HighShelfEq, NormalizationType, SegmentationStrategy, StreamEndCallback, Strictness, VoiceSelectMode
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.model_settings import ModelSettings, REGISTRY
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.reason_pauses import ReasonPauses, ReasonPauseTypes
from tts_audiobook_tool.util import *

import tts_audiobook_tool.app_types as app_types_module

app_types_module.PhraseGroup = PhraseGroup
BookSection.__annotations__['phrase_groups'] = list[PhraseGroup]

class Project(BaseModel):
    """
    Project settings. Changes remain in memory until ``save()`` is called explicitly.

    Project spec versions:
    - version 1: project text stored inline in `project.json`
    - version 2: project text stored externally in `project_text.json`
    - version 3: model-specific settings stored in model-keyed `model_settings`
      objects; version-2 flat fields are converted on load
    - version 4: one project-wide ordered voice_references list

    On save, `version` is always normalized to `PROJECT_SPEC_VERSION`.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        populate_by_name=True,
        # Allow model_settings with older Pydantic versions that protect all model_* names.
        protected_namespaces=("model_validate", "model_dump"),
    )

    _sound_segments: Any = PrivateAttr(default=None)
    _on_stream_end: StreamEndCallback | None = PrivateAttr(default=None)

    @property
    def sound_segments(self):
        return self._sound_segments

    @property
    def phrase_groups(self) -> list[PhraseGroup]:
        return self.book.phrase_groups

    @phrase_groups.setter
    def phrase_groups(self, value: list[PhraseGroup]) -> None:
        settings = self.book.segmentation_settings
        self.book = Book(
            sections=[BookSection(phrase_groups=value)],
            title=self.book.title,
            text_source_kind=self.book.text_source_kind or "legacy_flat",
            audio_source_kind=self.book.audio_source_kind or "unknown",
            segmentation_settings=settings,
        )

    @property
    def on_stream_end(self) -> StreamEndCallback | None:
        return self._on_stream_end

    @on_stream_end.setter
    def on_stream_end(self, value: StreamEndCallback | None) -> None:
        self._on_stream_end = value

    def get_high_shelf(self) -> HighShelfEq:
        return HighShelfEq.get_by_id(self.high_shelf) or HighShelfEq.DISABLED

    def has_multiple_book_sections(self) -> bool:
        return len(self.book.sections) > 1

    def can_use_bookmark_section_markers(self) -> bool:
        return not self.has_multiple_book_sections()

    def get_validated_chapter_mode(self, value: SectionMarkerMode | None = None) -> SectionMarkerMode:
        chapter_mode = value or self.chapter_mode
        if self.has_multiple_book_sections() and chapter_mode == SectionMarkerMode.BOOKMARKS:
            return SectionMarkerMode.FILES
        return chapter_mode

    def normalize_chapter_mode(self) -> bool:
        valid_mode = self.get_validated_chapter_mode()
        if valid_mode == self.chapter_mode:
            return False
        super().__setattr__('chapter_mode', valid_mode)
        return True

    # --- Fields ---

    dir_path: str = ""
    version: int = PROJECT_SPEC_VERSION
    # Persist the opaque catalog ID, including IDs unknown to this installation.
    tts_model_type: str = "none"
    model_settings: ModelSettings = Field(default_factory=ModelSettings)
    voice_references: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("voice_references", mode="before")
    @classmethod
    def _validate_voice_references(cls, value: Any) -> list[dict[str, str]]:
        from tts_audiobook_tool.project_support.voice_reference_migration import normalize_voice_references
        return normalize_voice_references(value)

    def get_tts_model_type(self) -> TtsModelType:
        """Resolve the saved selection without changing it or auto-detecting."""
        return TtsModelType.get_by_id(self.tts_model_type)

    @field_validator("model_settings", mode="before")
    @classmethod
    def _reconcile_model_settings(cls, value: Any) -> ModelSettings:
        """Validate and prune supplied objects; retain whole unknown objects."""
        if isinstance(value, ModelSettings):
            return value
        return REGISTRY.reconcile(value)

    def get_model_setting(self, model_id: str, name: str) -> Any:
        """Read a declared setting; legacy voice access reads the project list."""
        binding = REGISTRY.get(model_id, name)
        if binding.section == "voice_references":
            return [ref.get(name, "") for ref in self.voice_references]
        return REGISTRY.resolve(self.model_settings, binding)

    def set_model_setting(self, model_id: str, name: str, value: Any, *, reset: bool = False) -> None:
        """Write an override; legacy voice access never creates scoped copies."""
        binding = REGISTRY.get(model_id, name)
        if binding.section == "voice_references":
            value = [] if reset else value
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{model_id}.{name}: expected a list of strings")
            if name == "transcript":
                self.voice_references = [
                    {"file_name": ref["file_name"], "transcript": value[i] if i < len(value) else ""}
                    for i, ref in enumerate(self.voice_references)
                ]
            else:
                from tts_audiobook_tool.project_support.voice_reference_migration import normalize_voice_references
                self.voice_references = normalize_voice_references([
                    {"file_name": item, "transcript": self.voice_references[i].get("transcript", "") if i < len(self.voice_references) else ""}
                    for i, item in enumerate(value)
                ])
            return
        if reset or (binding.has_sentinel and value == binding.sentinel):
            REGISTRY.assign(self.model_settings, binding, value, reset=True)
        else:
            REGISTRY.assign(self.model_settings, binding, value)

    language_code: str = PROJECT_DEFAULT_LANGUAGE

    book: Book = Field(default_factory=lambda: Book(sections=[]))
    segmentation_strategy: SegmentationStrategy = PROJECT_DEFAULT_SEGMENTATION_STRATEGY
    max_words: int = MAX_WORDS_PER_SEGMENT_DEFAULT
    dialog_segmentation: bool = False
    word_substitutions: dict[str, str] = Field(default_factory=dict)

    # Generation-range sentinels:
    # - empty string means "all items" for compatibility reasons
    # - string literal "none" means no selection
    generate_range_string: str = Field(default="", alias="generate_range")
    
    marker_indices: set[int] = Field(default_factory=set, alias="markers")
    subdivide_phrases: bool = False
    export_type: ExportType = list(ExportType)[0]
    use_break_sound_effect: bool = Field(default=PROJECT_DEFAULT_BREAK_EFFECT, alias="use_section_sound_effect")
    normalization_type: NormalizationType = list(NormalizationType)[0]
    high_shelf: str = HighShelfEq.DISABLED.id
    reason_pauses: ReasonPauses = ReasonPauseTypes.default().value
    use_upsampler: bool = False
    realtime_save: bool = PROJECT_DEFAULT_REALTIME_SAVE
    realtime_line_range: tuple[int, int] | None = None
    limit_silence_gaps: bool = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS
    limit_silence_gaps_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION
    gen_auto_concat: bool = PROJECT_DEFAULT_GEN_AUTO_CONCAT
    streaming_chat: bool = PROJECT_DEFAULT_STREAMING_CHAT
    
    # Rem, UI nomenclature for this is "tolerance"
    strictness: Strictness = list(Strictness)[0]
    
    max_retries: int = PROJECT_MAX_RETRIES_DEFAULT
    chapter_mode: SectionMarkerMode = list(SectionMarkerMode)[0]
    voice_select_mode: VoiceSelectMode = VoiceSelectMode.get_default()

    # Placeholder attribute used when no TTS model exists
    none_voice_file_name: str = "" 

    def model_post_init(self, __context: Any) -> None:
        # A `dir_path` written by another operating system's path grammar is not
        # a directory here. Creating it anyway would litter the working
        # directory with a folder literally named `C:\Users\…`.
        if self.dir_path and not path_norm.looks_foreign(self.dir_path):
            ss_path = os.path.join(self.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
            if not os.path.exists(ss_path):
                try:
                    os.mkdir(ss_path)
                except Exception:
                    pass

        from tts_audiobook_tool.project_support.project_sound_segments import ProjectSoundSegments
        self._sound_segments = ProjectSoundSegments(self)

        # A Pocket preset can coexist with shared samples and takes precedence.

    @property
    def markers(self) -> set[int]:
        """Return a copy of the positive section-marker indices."""
        return set(self.marker_indices)

    @markers.setter
    def markers(self, value: set[int]) -> None:
        """Store positive marker indices, silently discarding non-positive values."""
        self.marker_indices = {marker for marker in value if marker > 0}

    @model_validator(mode='before')
    @classmethod
    def _normalize_loaded_project_dict(cls, d: Any, info: ValidationInfo) -> Any:
        warnings: list[str] | None = None
        if isinstance(info.context, dict):
            candidate = info.context.get('warnings')
            if isinstance(candidate, list):
                warnings = candidate
        return ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=warnings)

    @property
    def project_text_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_TEXT_FILE_NAME)

    def save(self) -> str:
        
        file_path = os.path.join(self.dir_path, PROJECT_JSON_FILE_NAME)

        def make_payload() -> dict:
            super(Project, self).__setattr__('version', PROJECT_SPEC_VERSION)
            self.normalize_chapter_mode()
            return ProjectSerializationUtil.to_project_json_dict(self)

        err = JsonSaveUtil.save(
            JsonArtifactType.PROJECT,
            file_path,
            make_payload,
        )
        if err:
            printt(f"\n{COL_ERROR}{err}\n")
            return err

        L.d(f"Saved {PROJECT_JSON_FILE_NAME}: {file_path}")
        return ""

    @property
    def sound_segments_path(self) -> str:
        if not self.dir_path:
            return ""  # TODO smth abt project not yet having a dir_path, etc
        return os.path.join(self.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

    @property
    def concat_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_CONCAT_SUBDIR)

    @property
    def realtime_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_REALTIME_OUTPUT_SUBDIR)

    def kill(self) -> None:
        self.sound_segments.observer.stop()
