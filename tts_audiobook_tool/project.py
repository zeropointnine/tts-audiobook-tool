from __future__ import annotations
import os
import threading
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from tts_audiobook_tool.app_types import Book, BookSection, SectionMarkerMode, ExportType, HighShelfEq, NormalizationType, SegmentationStrategy, StreamEndCallback, Strictness
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.util import *

import tts_audiobook_tool.app_types as app_types_module

app_types_module.PhraseGroup = PhraseGroup
BookSection.__annotations__['phrase_groups'] = list[PhraseGroup]

# Thread-local warning sink used during project load/normalization by
# project_load_util and project_serialization_util.
_tl = threading.local()


class Project(BaseModel):
    """
    Project settings. Assigning any field automatically saves to disk once `_autosave`
    is enabled (after loading). Use `with project.batch():` to make several assignments
    and save exactly once at the end.

    Project spec versions:
    - version 1: project text stored inline in `project.json`
    - version 2: project text stored externally in `project_text.json`

    On save, `version` is always normalized to `CURRENT_PROJECT_VERSION`.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        populate_by_name=True,
    )

    _autosave: bool = PrivateAttr(default=False)
    _phrase_groups_dirty: bool = PrivateAttr(default=False)
    _phrase_groups_inline_source: str = PrivateAttr(default="")
    _sound_segments: Any = PrivateAttr(default=None)
    _on_stream_end: StreamEndCallback | None = PrivateAttr(default=None)

    @property
    def sound_segments(self):
        return self._sound_segments

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

    language_code: str = PROJECT_DEFAULT_LANGUAGE

    book: Book = Field(default_factory=lambda: Book(sections=[]))
    phrase_groups: list[PhraseGroup] = Field(default_factory=list)

    segmentation_strategy: SegmentationStrategy = PROJECT_DEFAULT_SEGMENTATION_STRATEGY
    max_words: int = MAX_WORDS_PER_SEGMENT_DEFAULT
    word_substitutions: dict[str, str] = Field(default_factory=dict)

    # The segmentation strategy used to create the PhraseGroups from the source text
    applied_strategy: SegmentationStrategy | None = None
    # The max words per segment value used to create the PhraseGroups from the source text
    applied_max_words: int = 0
    # The language code used to create the PhraseGroups from the source text (ie, for pysbd)
    applied_language_code: str = ""

    generate_range_string: str = Field(default="", alias="generate_range")
    markers: list[int] = Field(default_factory=list, alias="markers")
    subdivide_phrases: bool = False
    export_type: ExportType = list(ExportType)[0]
    use_break_sound_effect: bool = Field(default=PROJECT_DEFAULT_BREAK_EFFECT, alias="use_section_sound_effect")
    normalization_type: NormalizationType = list(NormalizationType)[0]
    high_shelf: str = HighShelfEq.DISABLED.id
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

    # Placeholder attribute used when no TTS model exists
    none_voice_file_name: str = "" 

    oute_voice_file_name: str = ""
    oute_voice_json: dict = Field(default_factory=dict)
    oute_temperature: float = -1

    chatterbox_type: ChatterboxType = list(ChatterboxType)[0]
    chatterbox_voice_file_name: str = ""
    chatterbox_temperature: float = -1
    chatterbox_cfg: float = -1
    chatterbox_exaggeration: float = -1
    chatterbox_top_p: float = -1
    chatterbox_turbo_top_k: int = -1
    chatterbox_ml_repetition_penalty: float = -1
    chatterbox_turbo_repetition_penalty: float = -1
    chatterbox_seed: int = -1

    fish_s1_voice_file_name: str = ""
    fish_s1_voice_transcript: str = Field(default="", alias="fish_s1_voice_text")
    fish_s1_compile_enabled: bool = FishS1BaseModel.DEFAULT_COMPILE_ENABLED
    fish_s1_temperature: float = -1
    fish_s1_top_p: float = -1  # rem, s1 lib api has no top_k option (but s2 lib does)
    fish_s1_repetition_penalty: float = -1
    fish_s1_seed: int = -1

    fish_s2_voice_file_name: str = ""
    fish_s2_voice_transcript: str = Field(default="", alias="fish_s2_voice_text")
    fish_s2_compile_enabled: bool = FishS2BaseModel.DEFAULT_COMPILE_ENABLED
    fish_s2_temperature: float = -1
    fish_s2_top_p: float = -1
    fish_s2_top_k: int = -1
    fish_s2_seed: int = -1

    higgs_voice_file_name: str = ""
    higgs_voice_transcript: str = Field(default="", alias="higgs_voice_text")
    higgs_temperature: float = -1
    higgs_top_k: int = -1
    higgs_top_p: float = -1
    higgs_seed: int = -1

    vibevoice_voice_file_name: str = ""
    vibevoice_target: str = ""
    vibevoice_lora_target: str = Field(default="", alias="vibevoice_lora_path")
    vibevoice_cfg: float = -1
    vibevoice_steps: int = -1
    vibevoice_batch_size: int = 1
    vibevoice_seed: int = -1

    indextts2_temperature: float = -1
    indextts2_use_fp16: bool = IndexTts2BaseModel.DEFAULT_USE_FP16
    indextts2_voice_file_name: str = ""
    indextts2_emo_alpha: float = -1
    indextts2_emo_voice_file_name: str = ""
    indextts2_emo_vector: list[float] = Field(default_factory=list)
    indextts2_top_p: float = -1
    indextts2_top_k: int = -1
    indextts2_seed: int = -1

    glm_voice_file_name: str = ""
    glm_voice_transcript: str = Field(default="", alias="glm_voice_text")
    glm_sr: int = GlmBaseModel.SAMPLE_RATES[0]
    glm_seed: int = -1

    mira_voice_file_name: str = ""
    mira_temperature: float = -1
    mira_top_p: float = -1
    mira_top_k: int = -1
    mira_repetition_penalty: float = -1
    mira_batch_size: int = 1
    mira_seed: int = -1

    qwen3_target: str = ""
    qwen3_model_type: str = ""
    qwen3_voice_file_name: str = ""
    qwen3_voice_transcript: str = Field(default="", alias="qwen3_voice_text")
    qwen3_speaker_id: str = ""
    qwen3_instructions: str = ""
    qwen3_batch_size: int = 1
    qwen3_temperature: float = -1
    qwen3_top_k: int = -1
    qwen3_top_p: float = -1
    qwen3_repetition_penalty: float = -1
    qwen3_seed: int = -1

    pocket_voice_file_name: str = ""
    pocket_predefined_voice: str = ""
    pocket_model_code: str = ""
    pocket_temperature: float = -1
    pocket_seed: int = -1

    omnivoice_voice_file_name: str = ""
    omnivoice_voice_transcript: str = ""
    omnivoice_target: str = ""
    omnivoice_instruct: str = ""
    omnivoice_cfg: float = -1
    omnivoice_speed: float = -1
    omnivoice_num_step: int = -1
    omnivoice_seed: int = -1

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name in ('book', 'phrase_groups') and not name.startswith('_'):
            self._phrase_groups_dirty = True
        if name != '_autosave' and getattr(self, '_autosave', False):
            self.save()

    @contextmanager
    def batch(self):
        """Suppress per-assignment saves; save exactly once when the block exits."""
        prev = getattr(self, '_autosave', False)
        self._autosave = False
        try:
            yield
        finally:
            self._autosave = prev
            self.save()

    def model_post_init(self, __context: Any) -> None:
        if self.dir_path:
            ss_path = os.path.join(self.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
            if not os.path.exists(ss_path):
                try:
                    os.mkdir(ss_path)
                except Exception:
                    pass

        from tts_audiobook_tool.project_support.project_sound_segments import ProjectSoundSegments
        self._sound_segments = ProjectSoundSegments(self)

        if self.pocket_voice_file_name and self.pocket_predefined_voice:
            self.pocket_predefined_voice = ""

    @model_validator(mode='before')
    @classmethod
    def _normalize_loaded_project_dict(cls, d: Any) -> Any:
        return ProjectSerializationUtil.normalize_loaded_project_dict(d)

    @property
    def project_text_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_TEXT_FILE_NAME)

    def save(self, force_phrase_groups: bool=False) -> str:
        
        file_path = os.path.join(self.dir_path, PROJECT_JSON_FILE_NAME)
        
        try:
            # Ensure project version is up-to-date
            super().__setattr__('version', PROJECT_SPEC_VERSION)
            self.normalize_chapter_mode()
            
            with open(file_path, "w", encoding="utf-8") as file:
                import json
                json.dump(ProjectSerializationUtil.to_project_json_dict(self), file, indent=4)
            L.d(f"Saved {PROJECT_JSON_FILE_NAME}: {file_path}")

        except Exception as e:
            err = make_error_string(e)
            printt(f"\n{COL_ERROR}{err}\n")
            return err

        if force_phrase_groups or self._phrase_groups_dirty:
            return ProjectTextIOUtil.save_phrase_groups(self)
        
        return ""

    def set_oute_voice_and_save(self, voice_dict: dict, dest_file_stem: str) -> None:
        file_name = dest_file_stem + ".json"
        err = save_json(voice_dict, os.path.join(self.dir_path, file_name))
        if err:
            from tts_audiobook_tool import ask
            ask.ask_error(err)
            return
        with self.batch():
            self.oute_voice_file_name = file_name
            self.oute_voice_json = voice_dict

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
