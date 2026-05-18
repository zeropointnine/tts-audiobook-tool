from __future__ import annotations
import json
import os
import threading
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from tts_audiobook_tool.app_types import SectionMarkerMode, ExportType, HighShelfEq, NormalizationType, SegmentationStrategy, Sound, StreamEndCallback, Strictness
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

from tts_audiobook_tool.tts import Tts


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

    # --- Fields ---

    dir_path: str = ""
    version: int = PROJECT_SPEC_VERSION

    language_code: str = PROJECT_DEFAULT_LANGUAGE

    phrase_groups: list[PhraseGroup] = Field(default_factory=list)

    segmentation_strategy: SegmentationStrategy = list(SegmentationStrategy)[0]
    max_words: int = MAX_WORDS_PER_SEGMENT_DEFAULT
    word_substitutions: dict[str, str] = Field(default_factory=dict)

    # The segmentation strategy used to create the PhraseGroups from the source text
    applied_strategy: SegmentationStrategy | None = None
    # The max words per segment value used to create the PhraseGroups from the source text
    applied_max_words: int = 0
    # The language code used to create the PhraseGroups from the source text (ie, for pysbd)
    applied_language_code: str = ""

    generate_range_string: str = Field(default="", alias="generate_range")
    section_dividers: list[int] = Field(default_factory=list, alias="chapter_indices")
    subdivide_phrases: bool = False
    export_type: ExportType = list(ExportType)[0]
    use_section_sound_effect: bool = PROJECT_DEFAULT_SECTION_SOUND_EFFECT
    normalization_type: NormalizationType = list(NormalizationType)[0]
    high_shelf: str = HighShelfEq.DISABLED.id
    use_upsampler: bool = False
    realtime_save: bool = PROJECT_DEFAULT_REALTIME_SAVE
    realtime_line_range: tuple[int, int] | None = None
    limit_silence_gaps: bool = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS
    limit_silence_gaps_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION
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
        if name == 'phrase_groups' and not name.startswith('_'):
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

        from tts_audiobook_tool.project_sound_segments import ProjectSoundSegments
        self._sound_segments = ProjectSoundSegments(self)

        if self.pocket_voice_file_name and self.pocket_predefined_voice:
            self.pocket_predefined_voice = ""

    @model_validator(mode='before')
    @classmethod
    def _parse_raw(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return d

        use_tl_warnings = getattr(_tl, 'warnings', None) is not None

        def add_warning(attr_name: str, defaulting_to: Any) -> None:
            s = f"{COL_ERROR}Warning/info: {COL_DEFAULT}Missing or invalid value for: {COL_ACCENT}{attr_name}{COL_DEFAULT}\n"
            s += "This can occur if a new project property has been added to the app since the last time you opened this project.\n"
            s += f"Setting to default: {defaulting_to}"
            if use_tl_warnings:
                _tl.warnings.append(s)

        from tts_audiobook_tool.project_util import ProjectUtil

        ProjectUtil.remap_legacy_keys(d)

        # version
        value = d.get('version', 1)
        if not isinstance(value, int) or value < 1:
            value = 1
        d['version'] = value

        # text / text_segments → phrase_groups
        if "text" in d:
            lst = d.pop("text")
            result = PhraseGroup.phrase_groups_from_json_list(lst)
            if isinstance(result, str):
                printt(f"{COL_ERROR}Error loading project text: {result}")
                d['phrase_groups'] = []
            else:
                d['phrase_groups'] = result
        elif "text_segments" in d:
            lst = d.pop("text_segments")
            result = Phrase.phrases_from_json_dicts(lst)
            if isinstance(result, str):
                printt(f"{COL_ERROR}Error loading project text legacy format: {result}")
                printt()
                d['phrase_groups'] = []
            else:
                phrases = result
                phrase_groups = []
                for i in range(len(phrases)):
                    reason = phrases[i + 1].reason if i < len(phrases) - 1 else Reason.UNDEFINED
                    phrase_groups.append(PhraseGroup([Phrase(phrases[i].text, reason)]))
                d['phrase_groups'] = phrase_groups

        # word_substitutions_json_string → word_substitutions dict
        if 'word_substitutions_json_string' in d:
            s = d.pop('word_substitutions_json_string')
            result = ProjectUtil.parse_word_substitutions_json_string(s)
            d['word_substitutions'] = {} if isinstance(result, str) else result

        # segmentation_strategy
        s = d.get('segmentation_strategy', '')
        d['segmentation_strategy'] = SegmentationStrategy.from_id(s) or list(SegmentationStrategy)[0]

        # max_words
        i = d.get('max_words', MAX_WORDS_PER_SEGMENT_DEFAULT)
        if not isinstance(i, (int, float)):
            i = MAX_WORDS_PER_SEGMENT_DEFAULT
        i = int(i)
        if not (MAX_WORDS_PER_SEGMENT_MIN <= i <= MAX_WORDS_PER_SEGMENT_MAX):
            i = MAX_WORDS_PER_SEGMENT_DEFAULT
        d['max_words'] = i

        # applied_strategy
        s = d.get('applied_strategy', '')
        d['applied_strategy'] = SegmentationStrategy.from_id(s)  # None is valid

        # applied_max_words
        i = d.get('applied_max_words', 0)
        if not (i >= 0):
            i = 0
        d['applied_max_words'] = i

        # generate_range → normalize sentinels
        s = d.get('generate_range', '')
        if s in ('all', 'a'):
            d['generate_range'] = ''

        # chapter_indices → validate against phrase_groups length
        if 'chapter_indices' in d:
            phrase_groups = d.get('phrase_groups', [])
            lst = d['chapter_indices']
            if isinstance(lst, list):
                is_valid = all(
                    isinstance(idx, int) and 0 <= idx <= len(phrase_groups)
                    for idx in lst
                )
                if not is_valid:
                    printt(f"File cut points invalid: {lst}")
                    d['chapter_indices'] = []
            else:
                d['chapter_indices'] = []

        # export_type
        s = d.get('export_type', '')
        d['export_type'] = ExportType.get_by_id(s) or list(ExportType)[0]

        # normalization_type
        s = d.get('normalization_type', '')
        d['normalization_type'] = NormalizationType.from_id(s) or list(NormalizationType)[0]

        # high_shelf
        s = d.get('high_shelf', HighShelfEq.DISABLED.id)
        value = HighShelfEq.get_by_id(s)
        if value is None:
            value = HighShelfEq.DISABLED
            add_warning('high_shelf', value.id)
        d['high_shelf'] = value.id

        # use_usampler
        value = d.get('use_upsampler', False)
        if not isinstance(value, bool):
            value = False
        d['use_upsampler'] = value

        # realtime_line_range
        value = d.get('realtime_line_range', None)
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, int) for item in value):
            if value[0] == 0 and value[1] == 0:
                d['realtime_line_range'] = None
            else:
                d['realtime_line_range'] = (value[0], value[1])
        else:
            d['realtime_line_range'] = None

        # streaming_chat
        value = d.get('streaming_chat', None)
        if not isinstance(value, bool):
            value = True
            add_warning('streaming_chat', value)
        d['streaming_chat'] = value

        # limit_silence_gaps
        value = d.get('limit_silence_gaps', None)
        if not isinstance(value, bool):
            value = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS
            add_warning('limit_silence_gaps', value)
        d['limit_silence_gaps'] = value

        # limit_silence_gaps_duration
        value = d.get('limit_silence_gaps_duration', None)
        if not isinstance(value, (int, float)) or value <= 0:
            value = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION
            add_warning('limit_silence_gaps_duration', value)
        d['limit_silence_gaps_duration'] = float(value)

        # strictness (default depends on language_code)
        s = d.get('strictness', '')
        strictness = Strictness.get_by_id(s)
        if strictness is None:
            strictness = Strictness.get_recommended_default(d.get('language_code', ''))
            add_warning('strictness', strictness)
        d['strictness'] = strictness

        # max_retries
        value = d.get('max_retries', None)
        if not isinstance(value, int) or not (PROJECT_MAX_RETRIES_MIN <= value <= PROJECT_MAX_RETRIES_MAX):
            add_warning('max_retries', PROJECT_MAX_RETRIES_DEFAULT)
            value = PROJECT_MAX_RETRIES_DEFAULT
        d['max_retries'] = value

        # chapter_mode
        s = d.get('chapter_mode', '')
        value = SectionMarkerMode.get_by_id(s)
        if value is None:
            value = list(SectionMarkerMode)[0]
            add_warning('chapter_mode', value)
        d['chapter_mode'] = value

        # chatterbox_type
        s = d.get('chatterbox_type', '')
        chatterbox_type = ChatterboxType.get_by_id(s)
        if not chatterbox_type:
            chatterbox_type = list(ChatterboxType)[0]
            try:
                if Tts.get_type() == TtsModelInfos.CHATTERBOX:
                    add_warning('chatterbox_type', chatterbox_type.id)
            except AttributeError:
                pass  # Tts not yet initialized
        d['chatterbox_type'] = chatterbox_type

        # chatterbox_seed
        seed = d.get('chatterbox_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('chatterbox_seed', -1)
            seed = -1
        d['chatterbox_seed'] = seed

        # fish_s1_compile_enabled
        value = d.get('fish_s1_compile_enabled', None)
        if not isinstance(value, bool):
            value = True  # legacy compat
        d['fish_s1_compile_enabled'] = value

        # fish_s1_seed
        seed = d.get('fish_s1_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('fish_s1_seed', -1)
            seed = -1
        d['fish_s1_seed'] = seed

        # fish_s2_compile_enabled
        value = d.get('fish_s2_compile_enabled', None)
        if not isinstance(value, bool):
            value = True  # legacy compat
        d['fish_s2_compile_enabled'] = value

        # fish_s2_seed
        seed = d.get('fish_s2_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('fish_s2_seed', -1)
            seed = -1
        d['fish_s2_seed'] = seed

        # higgs_seed
        seed = d.get('higgs_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('higgs_seed', -1)
            seed = -1
        d['higgs_seed'] = int(seed)

        # vibevoice_batch_size
        value = d.get('vibevoice_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('vibevoice_batch_size', value)
            value = int(value)
        d['vibevoice_batch_size'] = value

        # vibevoice_seed
        seed = d.get('vibevoice_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('vibevoice_seed', -1)
            seed = -1
        d['vibevoice_seed'] = seed

        # indextts2_emo_vector
        o = d.get('indextts2_emo_vector', [])
        if not isinstance(o, list):
            o = []
        d['indextts2_emo_vector'] = o

        # indextts2_seed
        seed = d.get('indextts2_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('indextts2_seed', -1)
            seed = -1
        d['indextts2_seed'] = int(seed)

        # glm_sr
        sr = d.get('glm_sr', 0)
        if sr not in GlmBaseModel.SAMPLE_RATES:
            sr = GlmBaseModel.SAMPLE_RATES[0]
        d['glm_sr'] = sr

        # glm_seed
        seed = d.get('glm_seed', -1)
        if not isinstance(seed, (int, float)) or not (seed >= -1):
            seed = -1
        d['glm_seed'] = int(seed)

        # mira_temperature
        value = d.get('mira_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (MiraBaseModel.TEMPERATURE_MIN <= value <= MiraBaseModel.TEMPERATURE_MAX):
                value = -1
                add_warning('mira_temperature', value)
        d['mira_temperature'] = value

        # mira_batch_size
        value = d.get('mira_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('mira_batch_size', value)
            value = int(value)
        d['mira_batch_size'] = value

        # mira_seed
        seed = d.get('mira_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('mira_seed', -1)
            seed = -1
        d['mira_seed'] = int(seed)

        # qwen3_temperature
        value = d.get('qwen3_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (Qwen3BaseModel.TEMPERATURE_MIN <= value <= Qwen3BaseModel.TEMPERATURE_MAX):
                value = -1
                add_warning('qwen3_temperature', value)
        d['qwen3_temperature'] = value

        # qwen3_batch_size
        value = d.get('qwen3_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('qwen3_batch_size', value)
            value = int(value)
        d['qwen3_batch_size'] = value

        # qwen3_seed
        seed = d.get('qwen3_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('qwen3_seed', -1)
            seed = -1
        d['qwen3_seed'] = int(seed)

        # pocket_temperature
        from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
        value = d.get('pocket_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (PocketBaseModel.TEMPERATURE_MIN <= value <= PocketBaseModel.TEMPERATURE_MAX):
                value = -1
                add_warning('pocket_temperature', value)
        d['pocket_temperature'] = value

        # pocket_seed
        seed = d.get('pocket_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('pocket_seed', -1)
            seed = -1
        d['pocket_seed'] = int(seed)

        d['pocket_model_code'] = d.get('pocket_model_code', '')

        # omnivoice_seed
        seed = d.get("omnivoice_seed", -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning("omnivoice_seed", -1)
            seed = -1
        d["omnivoice_seed"] = int(seed)

        # omnivoice_num_step
        if "omnivoice_num_step" not in d and "omnivoice_steps" in d:
            d["omnivoice_num_step"] = d.get("omnivoice_steps", -1)
        value = d.get("omnivoice_num_step", -1)
        if value != -1:
            if not isinstance(value, int) or not (OmniVoiceBaseModel.MIN_STEPS <= value <= OmniVoiceBaseModel.MAX_STEPS):
                value = -1
                add_warning("omnivoice_num_step", value)
        d["omnivoice_num_step"] = int(value)

        # omnivoice_speed
        value = d.get("omnivoice_speed", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (0.5 <= value <= 2.0):
                value = -1
                add_warning("omnivoice_speed", value)
        d["omnivoice_speed"] = value

        # omnivoice_cfg
        value = d.get("omnivoice_cfg", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (OmniVoiceBaseModel.CFG_MIN <= value <= OmniVoiceBaseModel.CFG_MAX):
                value = -1
                add_warning("omnivoice_cfg", value)
        d["omnivoice_cfg"] = value

        return d

    def to_dict(self) -> dict:
        return {
            "dir_path": self.dir_path,
            "version": self.version,

            "language_code": self.language_code,

            "segmentation_strategy": self.segmentation_strategy.id,
            "max_words": self.max_words,
            "word_substitutions_json_string": json.dumps(self.word_substitutions),
            "applied_language_code": self.applied_language_code,
            "applied_strategy": self.applied_strategy.id if self.applied_strategy else "",
            "applied_max_words": self.applied_max_words,

            "generate_range": self.generate_range_string,
            "chapter_indices": self.section_dividers,
            "subdivide_phrases": self.subdivide_phrases,
            "export_type": self.export_type.id,
            "use_section_sound_effect": self.use_section_sound_effect,
            "normalization_type": self.normalization_type.value.id,
            "high_shelf": self.high_shelf,
            "use_upsampler": self.use_upsampler,
            "realtime_save": self.realtime_save,
            "realtime_line_range": self.realtime_line_range,
            "limit_silence_gaps": self.limit_silence_gaps,
            "limit_silence_gaps_duration": self.limit_silence_gaps_duration,
            "streaming_chat": self.streaming_chat,
            "strictness": self.strictness.id,
            "max_retries": self.max_retries,
            "chapter_mode": self.chapter_mode.id,

            "none_voice_file_name": self.none_voice_file_name,

            "oute_voice_file_name": self.oute_voice_file_name,
            "oute_temperature": self.oute_temperature,

            "chatterbox_type": self.chatterbox_type.id,
            "chatterbox_voice_file_name": self.chatterbox_voice_file_name,
            "chatterbox_temperature": self.chatterbox_temperature,
            "chatterbox_top_p": self.chatterbox_top_p,
            "chatterbox_turbo_top_k": self.chatterbox_turbo_top_k,
            "chatterbox_ml_repetition_penalty": self.chatterbox_ml_repetition_penalty,
            "chatterbox_turbo_repetition_penalty": self.chatterbox_turbo_repetition_penalty,
            "chatterbox_cfg": self.chatterbox_cfg,
            "chatterbox_exaggeration": self.chatterbox_exaggeration,
            "chatterbox_seed": self.chatterbox_seed,

            "fish_s1_voice_file_name": self.fish_s1_voice_file_name,
            "fish_s1_voice_text": self.fish_s1_voice_transcript,
            "fish_s1_temperature": self.fish_s1_temperature,
            "fish_s1_top_p": self.fish_s1_top_p,
            "fish_s1_repetition_penalty": self.fish_s1_repetition_penalty,
            "fish_s1_seed": self.fish_s1_seed,
            "fish_s1_compile_enabled": self.fish_s1_compile_enabled,

            "fish_s2_voice_file_name": self.fish_s2_voice_file_name,
            "fish_s2_voice_text": self.fish_s2_voice_transcript,
            "fish_s2_temperature": self.fish_s2_temperature,
            "fish_s2_top_p": self.fish_s2_top_p,
            "fish_s2_top_k": self.fish_s2_top_k,
            "fish_s2_seed": self.fish_s2_seed,
            "fish_s2_compile_enabled": self.fish_s2_compile_enabled,

            "higgs_voice_file_name": self.higgs_voice_file_name,
            "higgs_voice_text": self.higgs_voice_transcript,
            "higgs_temperature": self.higgs_temperature,
            "higgs_top_k": self.higgs_top_k,
            "higgs_top_p": self.higgs_top_p,
            "higgs_seed": self.higgs_seed,

            "vibevoice_voice_file_name": self.vibevoice_voice_file_name,
            "vibevoice_target": self.vibevoice_target,
            "vibevoice_lora_path": self.vibevoice_lora_target,
            "vibevoice_cfg": self.vibevoice_cfg,
            "vibevoice_steps": self.vibevoice_steps,
            "vibevoice_batch_size": self.vibevoice_batch_size,
            "vibevoice_seed": self.vibevoice_seed,

            "indextts2_voice_file_name": self.indextts2_voice_file_name,
            "indextts2_temperature": self.indextts2_temperature,
            "indextts2_emo_voice_file_name": self.indextts2_emo_voice_file_name,
            "indextts2_emo_vector": self.indextts2_emo_vector,
            "indextts2_emo_alpha": self.indextts2_emo_alpha,
            "indextts2_use_fp16": self.indextts2_use_fp16,
            "indextts2_top_p": self.indextts2_top_p,
            "indextts2_top_k": self.indextts2_top_k,
            "indextts2_seed": self.indextts2_seed,

            "glm_voice_file_name": self.glm_voice_file_name,
            "glm_voice_text": self.glm_voice_transcript,
            "glm_sr": self.glm_sr,
            "glm_seed": self.glm_seed,

            "mira_voice_file_name": self.mira_voice_file_name,
            "mira_temperature": self.mira_temperature,
            "mira_top_p": self.mira_top_p,
            "mira_top_k": self.mira_top_k,
            "mira_repetition_penalty": self.mira_repetition_penalty,
            "mira_batch_size": self.mira_batch_size,
            "mira_seed": self.mira_seed,

            "qwen3_target": self.qwen3_target,
            "qwen3_model_type": self.qwen3_model_type,
            "qwen3_voice_file_name": self.qwen3_voice_file_name,
            "qwen3_voice_text": self.qwen3_voice_transcript,
            "qwen3_speaker_id": self.qwen3_speaker_id,
            "qwen3_instructions": self.qwen3_instructions,
            "qwen3_batch_size": self.qwen3_batch_size,
            "qwen3_temperature": self.qwen3_temperature,
            "qwen3_top_k": self.qwen3_top_k,
            "qwen3_top_p": self.qwen3_top_p,
            "qwen3_repetition_penalty": self.qwen3_repetition_penalty,
            "qwen3_seed": self.qwen3_seed,

            "pocket_voice_file_name": self.pocket_voice_file_name,
            "pocket_predefined_voice": self.pocket_predefined_voice,
            "pocket_model_code": self.pocket_model_code,
            "pocket_temperature": self.pocket_temperature,
            "pocket_seed": self.pocket_seed,

            "omnivoice_voice_file_name": self.omnivoice_voice_file_name,
            "omnivoice_voice_transcript": self.omnivoice_voice_transcript,
            "omnivoice_target": self.omnivoice_target,
            "omnivoice_instruct": self.omnivoice_instruct,
            "omnivoice_cfg": self.omnivoice_cfg,
            "omnivoice_speed": self.omnivoice_speed,
            "omnivoice_num_step": self.omnivoice_num_step,
            "omnivoice_seed": self.omnivoice_seed
        }

    def to_snapshot_dict(self) -> dict:
        """
        Returns a project settings snapshot suitable for embedding in ABR metadata.

        This intentionally includes `dir_path` as a best-effort hint for future
        import flows that may want to locate and optionally copy related project
        files such as voice clones.
        """
        d = self.to_dict()
        return d

    @property
    def project_text_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_TEXT_FILE_NAME)

    def phrase_groups_to_dict(self) -> dict:
        return {
            "format": "phrase_groups.v1",
            "phrase_groups": PhraseGroup.phrase_groups_to_json_list(self.phrase_groups)
        }

    def save_phrase_groups(self) -> str:
        file_path = self.project_text_path
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.phrase_groups_to_dict(), file, indent=4)
            self._phrase_groups_dirty = False
            self._phrase_groups_inline_source = ""
            L.d(f"Saved {PROJECT_TEXT_FILE_NAME}: {file_path}")
            return ""
        except Exception as e:
            err = make_error_string(e)
            printt(f"\n{COL_ERROR}{err}\n")
            return err

    def save(self, force_phrase_groups: bool=False) -> str:
        
        file_path = os.path.join(self.dir_path, PROJECT_JSON_FILE_NAME)
        
        try:
            # Ensure project version is up-to-date
            super().__setattr__('version', PROJECT_SPEC_VERSION)
            
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.to_dict(), file, indent=4)
            L.d(f"Saved {PROJECT_JSON_FILE_NAME}: {file_path}")

        except Exception as e:
            err = make_error_string(e)
            printt(f"\n{COL_ERROR}{err}\n")
            return err

        if force_phrase_groups or self._phrase_groups_dirty:
            return self.save_phrase_groups()
        
        return ""

    def set_phrase_groups_and_save(
            self,
            phrase_groups: list[PhraseGroup],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str
    ) -> None:

        with self.batch():
            self.phrase_groups = phrase_groups
            self.applied_strategy = strategy
            self.applied_max_words = max_words
            self.applied_language_code = language_code
            # Setting this invalidates some things
            self.section_dividers = []
            self.generate_range_string = ""
            self.realtime_line_range = None

        self.save_raw_text(raw_text)  # saved for reference

    def set_phrase_groups_chapters_and_save(
            self,
            phrase_groups: list[PhraseGroup],
            section_dividers: list[int],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str
    ) -> None:

        with self.batch():
            self.phrase_groups = phrase_groups
            self.applied_strategy = strategy
            self.applied_max_words = max_words
            self.applied_language_code = language_code
            self.section_dividers = section_dividers
            self.generate_range_string = ""
            self.realtime_line_range = None

        self.save_raw_text(raw_text)  # saved for reference

    def save_raw_text(self, raw_text: str) -> None:
        file_path = os.path.join(self.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_text)
        except Exception as e:
            L.e(f"Error saving raw text: {e}")  # TODO need to return error

    def load_raw_text(self) -> str:
        file_path = os.path.join(self.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            L.e(f"Error saving raw text: {e}")  # TODO need to return error
            return ""

    def set_voice_and_save(
            self,
            sound: Sound,
            voice_file_stem: str,
            transcript: str,
            tts_type: TtsModelInfos,
            is_secondary: bool=False
    ) -> str:
        """
        Saves voice sound file, and updates and saves project properties.
        Sound file is expected to already be post-processed (viz, resampled to target model's native sr).
        Returns error string on fail.
        """
        # Add "_modelname" to filename
        dest_file_name = f"{voice_file_stem}_{tts_type.value.file_tag}.flac"
        dest_path = Path(self.dir_path) / dest_file_name

        # Save flac file to project dir
        err = SoundFileUtil.save_flac(sound, str(dest_path))
        if err:
            return err

        info = tts_type.value
        voice_file_name_attr = info.voice_file_name_attr
        voice_transcript_attr = info.voice_transcript_attr

        # Update the correct voice file name property of project, and save
        with self.batch():
            if tts_type == TtsModelInfos.INDEXTTS2 and is_secondary: # special case
                self.indextts2_emo_voice_file_name = dest_file_name
            else:
                if not voice_file_name_attr:
                    raise Exception(f"Unsupported tts type {tts_type}")
                setattr(self, voice_file_name_attr, dest_file_name)

            if voice_transcript_attr:
                setattr(self, voice_transcript_attr, transcript)

            if tts_type == TtsModelInfos.POCKET:
                self.pocket_predefined_voice = ""
        return ""

    def set_oute_voice_and_save(self, voice_dict: dict, dest_file_stem: str) -> None:
        file_name = dest_file_stem + ".json"
        err = save_json(voice_dict, os.path.join(self.dir_path, file_name))
        if err:
            from tts_audiobook_tool.ask_util import AskUtil
            AskUtil.ask_error(err)
            return
        with self.batch():
            self.oute_voice_file_name = file_name
            self.oute_voice_json = voice_dict

    def clear_voice_and_save(self, tts_type: TtsModelInfos, is_secondary: bool=False) -> None:
        info = tts_type.value
        voice_file_name_attr = info.voice_file_name_attr
        voice_transcript_attr = info.voice_transcript_attr

        with self.batch():
            if tts_type == TtsModelInfos.INDEXTTS2 and is_secondary:
                self.indextts2_emo_voice_file_name = ""
            else:
                if not voice_file_name_attr:
                    raise ValueError(f"Unsupported tts_type: {tts_type}")
                setattr(self, voice_file_name_attr, "")

            if voice_transcript_attr:
                setattr(self, voice_transcript_attr, "")

            if tts_type == TtsModelInfos.POCKET:
                self.pocket_predefined_voice = ""

    @property
    def voice_label(self) -> str:
        """
        UI-facing voice clone label derived from filename
        """
        if Tts.get_type() == TtsModelInfos.POCKET:
            if self.pocket_predefined_voice:
                return self.pocket_predefined_voice
            value = self.pocket_voice_file_name
            if not value:
                return "none"
            return ellipsize_path_for_menu(value.removesuffix("_pocket.flac"))
        value = getattr(self, Tts.get_type().value.voice_file_name_attr, "")
        if not value:
            return "none"
        value = value.removesuffix(f"_{Tts.get_type().value.file_tag}.flac")
        return ellipsize_path_for_menu(value)

    @property
    def has_voice(self) -> bool:
        if Tts.get_type() == TtsModelInfos.POCKET:
            return bool(self.pocket_predefined_voice or self.pocket_voice_file_name)
        value = getattr(self, Tts.get_type().value.voice_file_name_attr, "")
        return bool(value)

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
        return os.path.join(self.dir_path, PROJECT_REALTIME_SUBDIR)

    def emo_vector_to_string(self) -> str:
        if not self.indextts2_emo_vector or sum(self.indextts2_emo_vector) == 0:
            return "none"
        strings = []
        for item in self.indextts2_emo_vector:
            string = f"{item:.1f}".replace(".0", "")  # either one or no decimal point
            strings.append(string)
        return ",".join(strings)

    def verify_voice_files_exist(self) -> bool:
        """
        Checks whether current-model voice clone files exist and are valid sound files.
        Invalid refs are cleared and warnings are printed.
        Returns True when one or more attributes were cleared.
        """
        model_type = Tts.get_type()
        info = model_type.value

        attribs: list[str] = []
        if info.voice_file_name_attr:
            attribs.append(info.voice_file_name_attr)

        if model_type == TtsModelInfos.INDEXTTS2:
            attribs.append("indextts2_emo_voice_file_name")

        # Oute special case: voice_file_name_attr points to JSON payload, not a sound file name
        attribs = [attrib for attrib in attribs if attrib.endswith("_voice_file_name")]

        if not attribs:
            return False

        warnings = []
        for attrib in attribs:
            file_name = getattr(self, attrib, "")
            if not file_name:
                continue

            file_path = os.path.join(self.dir_path, file_name)
            if not os.path.exists(file_path):
                warnings.append((attrib, file_name, "file not found"))
                setattr(self, attrib, "")
                continue

            err = SoundFileUtil.is_valid_sound_file(file_path)
            if err:
                warnings.append((attrib, file_name, err))
                setattr(self, attrib, "")

        if warnings:
            printt(f"{COL_ERROR}Warning/info: {COL_DEFAULT}Problem with saved voice clone file(s) for current model {COL_ACCENT}{info.ui['proper_name']}{COL_DEFAULT}")
            for attrib, file_name, reason in warnings:
                printt(f"- {COL_ACCENT}{attrib}{COL_DEFAULT}: {file_name}")
                printt(f"  {COL_DIM}{reason}{COL_DEFAULT}")
            printt("Clearing saved reference(s) and continuing.")
            printt()

        return bool(warnings)

    def kill(self) -> None:
        self.sound_segments.observer.stop()

    @property
    def batch_size(self) -> int:
        field = Tts.get_type().value.batch_size_project_field
        if not field:
            return 1
        if not hasattr(self, field):
            raise ValueError(f"Unrecognized attribute {field}")
        value = getattr(self, field)
        if value == -1:
            value = PROJECT_BATCH_SIZE_DEFAULT
        elif value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX  # clamp silently
        return value

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        field = Tts.get_type().value.batch_size_project_field
        if not field:
            raise ValueError(f"No support for batch_size for the current model")
        if not hasattr(self, field):
            raise ValueError(f"Unrecognized attribute {field}")
        if value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX  # clamp silently
        setattr(self, field, value)

    @property
    def is_language_cjk(self) -> bool:
        if self.language_code in ["zh", "ja", "ko"]:
            return True
        if self.language_code.startswith(("zh-", "ja-", "ko-")):
            return True
        return False


