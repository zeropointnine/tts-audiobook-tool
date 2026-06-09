from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tts_audiobook_tool.app_types import (
    Book,
    BookSection,
    BookSegmentationSettings,
    ExportType,
    HighShelfEq,
    NormalizationType,
    SectionMarkerMode,
    SegmentationStrategy,
    Strictness,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import printt

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class ProjectSerializationUtil:
    @staticmethod
    def normalize_loaded_project_dict(d: Any) -> Any:

        if not isinstance(d, dict):
            return d

        from tts_audiobook_tool import project as project_module

        tl = project_module._tl
        use_tl_warnings = getattr(tl, 'warnings', None) is not None

        def add_warning(attr_name: str, defaulting_to: Any) -> None:
            s = f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Missing or invalid value for {COL_ACCENT}{attr_name}{COL_DEFAULT}\n"
            s += "This can occur if a new project property or feature has been\n"
            s += "added to the app since the last time you opened this project.\n"
            s += f"Setting to default: {defaulting_to}"
            s += "\n"
            if use_tl_warnings:
                tl.warnings.append(s)

        def normalize_bool(key: str, default: bool, *, warn: bool=False) -> bool:
            value = d.get(key, None)
            if not isinstance(value, bool):
                value = default
                if warn:
                    add_warning(key, value)
            d[key] = value
            return value

        def normalize_int(
                key: str,
                default: int,
                *,
                min_value: int | None=None,
                max_value: int | None=None,
                warn: bool=False,
                allow_float: bool=True,
        ) -> int:
            raw_value = d.get(key, default)
            valid_types = (int, float) if allow_float else (int,)
            if not isinstance(raw_value, valid_types):
                value = default
                is_valid = False
            else:
                value = int(raw_value)
                is_valid = True

            if min_value is not None and value < min_value:
                is_valid = False
            if max_value is not None and value > max_value:
                is_valid = False

            if not is_valid:
                value = default
                if warn:
                    add_warning(key, value)

            d[key] = value
            return value

        def normalize_by_id(key: str, resolver, default, *, warn: bool=False):
            raw_value = d.get(key, '')
            value = resolver(raw_value)
            if value is None:
                value = default
                if warn:
                    add_warning(key, getattr(value, 'id', value))
            d[key] = value
            return value

        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
        from tts_audiobook_tool.project_support.project_util import ProjectUtil

        ProjectLoadUtil.remap_legacy_keys(d)

        if "omnivoice_num_step" not in d and "omnivoice_steps" in d:
            d["omnivoice_num_step"] = d.get("omnivoice_steps", -1)

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

        if 'word_substitutions_json_string' in d:
            s = d.pop('word_substitutions_json_string')
            result = ProjectUtil.parse_word_substitutions_json_string(s)
            d['word_substitutions'] = {} if isinstance(result, str) else result

        if 'markers' not in d and 'chapter_indices' in d:
            d['markers'] = d['chapter_indices']

        value = d.get('version', 1)
        if not isinstance(value, int) or value < 1:
            value = 1
        d['version'] = value

        normalize_by_id('segmentation_strategy', SegmentationStrategy.from_id, PROJECT_DEFAULT_SEGMENTATION_STRATEGY)

        normalize_int(
            'max_words',
            MAX_WORDS_PER_SEGMENT_DEFAULT,
            min_value=MAX_WORDS_PER_SEGMENT_MIN,
            max_value=MAX_WORDS_PER_SEGMENT_MAX,
        )

        s = d.get('applied_strategy', '')
        d['applied_strategy'] = SegmentationStrategy.from_id(s)

        i = d.get('applied_max_words', 0)
        if not (i >= 0):
            i = 0
        d['applied_max_words'] = i

        s = d.get('generate_range', '')
        if s in ('all', 'a'):
            d['generate_range'] = ''

        if 'markers' in d:
            phrase_groups = d.get('phrase_groups', [])
            if not phrase_groups and isinstance(d.get('book'), Book):
                phrase_groups = d['book'].phrase_groups
            lst = d['markers']
            if isinstance(lst, list):
                is_valid = all(
                    isinstance(idx, int) and 0 <= idx <= len(phrase_groups)
                    for idx in lst
                )
                if not is_valid:
                    printt(f"File cut points invalid: {lst}")
                    d['markers'] = []
            else:
                d['markers'] = []

        legacy_settings = BookSegmentationSettings(
            language_code=d.get('applied_language_code', ''),
            max_words_per_segment=d.get('applied_max_words', 0),
            strategy=d.get('applied_strategy') or BookSegmentationSettings().strategy,
        )
        book = d.get('book')
        if isinstance(book, Book):
            ProjectBookUtil.sync_parse_dict_legacy_segmentation_from_book(d)
        elif d.get('phrase_groups'):
            d['book'] = Book(
                sections=[BookSection(phrase_groups=d.get('phrase_groups', []))],
                segmentation_settings=legacy_settings,
                text_source_kind="legacy_flat",
                audio_source_kind="unknown",
            )

        normalize_by_id('export_type', ExportType.get_by_id, list(ExportType)[0])
        normalize_by_id('normalization_type', NormalizationType.from_id, list(NormalizationType)[0])

        s = d.get('high_shelf', HighShelfEq.DISABLED.id)
        value = HighShelfEq.get_by_id(s)
        if value is None:
            value = HighShelfEq.DISABLED
            add_warning('high_shelf', value.id)
        d['high_shelf'] = value.id

        normalize_bool('use_upsampler', False)

        value = d.get('realtime_line_range', None)
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, int) for item in value):
            if value[0] == 0 and value[1] == 0:
                d['realtime_line_range'] = None
            else:
                d['realtime_line_range'] = (value[0], value[1])
        else:
            d['realtime_line_range'] = None

        normalize_bool('streaming_chat', True, warn=True)
        normalize_bool('limit_silence_gaps', PROJECT_DEFAULT_LIMIT_SILENCE_GAPS, warn=True)

        value = d.get('limit_silence_gaps_duration', None)
        if not isinstance(value, (int, float)) or value <= 0:
            value = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION
            add_warning('limit_silence_gaps_duration', value)
        d['limit_silence_gaps_duration'] = float(value)

        normalize_bool('gen_auto_concat', PROJECT_DEFAULT_GEN_AUTO_CONCAT, warn=True)

        s = d.get('strictness', '')
        strictness = Strictness.get_by_id(s)
        if strictness is None:
            strictness = Strictness.get_recommended_default(d.get('language_code', ''))
            add_warning('strictness', strictness)
        d['strictness'] = strictness

        normalize_int(
            'max_retries',
            PROJECT_MAX_RETRIES_DEFAULT,
            min_value=PROJECT_MAX_RETRIES_MIN,
            max_value=PROJECT_MAX_RETRIES_MAX,
            warn=True,
            allow_float=False,
        )

        value = normalize_by_id('chapter_mode', SectionMarkerMode.get_by_id, list(SectionMarkerMode)[0], warn=True)

        book = d.get('book')
        if isinstance(book, Book) and len(book.sections) > 1 and value == SectionMarkerMode.BOOKMARKS:
            value = SectionMarkerMode.FILES

        d['chapter_mode'] = value

        s = d.get('chatterbox_type', '')
        chatterbox_type = ChatterboxType.get_by_id(s)
        if not chatterbox_type:
            chatterbox_type = list(ChatterboxType)[0]
            try:
                from tts_audiobook_tool.tts import Tts

                if Tts.get_type() == TtsModelType.CHATTERBOX:
                    add_warning('chatterbox_type', chatterbox_type.id)
            except AttributeError:
                pass
        d['chatterbox_type'] = chatterbox_type

        seed = d.get('chatterbox_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('chatterbox_seed', -1)
            seed = -1
        d['chatterbox_seed'] = seed

        normalize_bool('fish_s1_compile_enabled', True)

        seed = d.get('fish_s1_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('fish_s1_seed', -1)
            seed = -1
        d['fish_s1_seed'] = seed

        normalize_bool('fish_s2_compile_enabled', True)
        normalize_int('fish_s2_rolling_cont', 0, min_value=0, max_value=3, warn=True)
        normalize_int('fish_s2_server_concurrent_requests', 1, min_value=1, max_value=PROJECT_BATCH_SIZE_MAX, warn=True)

        seed = d.get('fish_s2_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('fish_s2_seed', -1)
            seed = -1
        d['fish_s2_seed'] = seed

        seed = d.get('higgs_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('higgs_seed', -1)
            seed = -1
        d['higgs_seed'] = int(seed)

        seed = d.get('higgs_v3_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('higgs_v3_seed', -1)
            seed = -1
        d['higgs_v3_seed'] = int(seed)

        value = d.get('higgs_v3_batch_size', PROJECT_BATCH_SIZE_DEFAULT)
        if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
            value = PROJECT_BATCH_SIZE_DEFAULT
            add_warning('higgs_v3_batch_size', value)
        d['higgs_v3_batch_size'] = int(value)

        value = d.get('vibevoice_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('vibevoice_batch_size', value)
            value = int(value)
        d['vibevoice_batch_size'] = value

        seed = d.get('vibevoice_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('vibevoice_seed', -1)
            seed = -1
        d['vibevoice_seed'] = seed

        o = d.get('indextts2_emo_vector', [])
        if not isinstance(o, list):
            o = []
        d['indextts2_emo_vector'] = o

        seed = d.get('indextts2_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('indextts2_seed', -1)
            seed = -1
        d['indextts2_seed'] = int(seed)

        sr = d.get('glm_sr', 0)
        if sr not in GlmBaseModel.SAMPLE_RATES:
            sr = GlmBaseModel.SAMPLE_RATES[0]
        d['glm_sr'] = sr

        seed = d.get('glm_seed', -1)
        if not isinstance(seed, (int, float)) or not (seed >= -1):
            seed = -1
        d['glm_seed'] = int(seed)

        value = d.get('mira_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (MiraBaseModel.TEMPERATURE_MIN <= value <= MiraBaseModel.TEMPERATURE_MAX):
                value = -1
                add_warning('mira_temperature', value)
        d['mira_temperature'] = value

        value = d.get('mira_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('mira_batch_size', value)
            value = int(value)
        d['mira_batch_size'] = value

        seed = d.get('mira_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('mira_seed', -1)
            seed = -1
        d['mira_seed'] = int(seed)

        value = d.get('moss_target', '')
        if not isinstance(value, str):
            value = ''
            add_warning('moss_target', value)
        d['moss_target'] = value
        moss_delay_config = MossConfigs.DELAY.value
        moss_local_config = MossConfigs.LOCAL.value

        value = d.get('moss_delay_temperature', d.get('moss_temperature', -1))
        if value != -1:
            if not isinstance(value, (float, int)) or not (moss_delay_config.temperature_min <= value <= moss_delay_config.temperature_max):
                value = -1
                add_warning('moss_delay_temperature', value)
        d['moss_delay_temperature'] = value

        value = d.get('moss_delay_top_p', d.get('moss_top_p', -1))
        if value != -1:
            if not isinstance(value, (float, int)) or not (moss_delay_config.audio_top_p_min <= value <= moss_delay_config.audio_top_p_max):
                value = -1
                add_warning('moss_delay_top_p', value)
        d['moss_delay_top_p'] = value

        value = d.get('moss_delay_top_k', d.get('moss_top_k', -1))
        if value != -1:
            if not isinstance(value, (float, int)) or not (moss_delay_config.audio_top_k_min <= value <= moss_delay_config.audio_top_k_max):
                value = -1
                add_warning('moss_delay_top_k', value)
            value = int(value)
        d['moss_delay_top_k'] = value

        value = d.get('moss_local_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (moss_local_config.temperature_min <= value <= moss_local_config.temperature_max):
                value = -1
                add_warning('moss_local_temperature', value)
        d['moss_local_temperature'] = value

        value = d.get('moss_local_top_p', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (moss_local_config.audio_top_p_min <= value <= moss_local_config.audio_top_p_max):
                value = -1
                add_warning('moss_local_top_p', value)
        d['moss_local_top_p'] = value

        value = d.get('moss_local_top_k', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (moss_local_config.audio_top_k_min <= value <= moss_local_config.audio_top_k_max):
                value = -1
                add_warning('moss_local_top_k', value)
            value = int(value)
        d['moss_local_top_k'] = value

        value = d.get('moss_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('moss_batch_size', value)
            value = int(value)
        d['moss_batch_size'] = value

        normalize_int('moss_rolling_cont', 0, min_value=0, warn=True)

        seed = d.get('moss_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('moss_seed', -1)
            seed = -1
        d['moss_seed'] = int(seed)

        normalize_int('qwen3_rolling_cont', 0, min_value=0, warn=True)

        value = d.get('qwen3_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (Qwen3BaseModel.TEMPERATURE_MIN <= value <= Qwen3BaseModel.TEMPERATURE_MAX):
                value = -1
                add_warning('qwen3_temperature', value)
        d['qwen3_temperature'] = value

        value = d.get('qwen3_batch_size', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning('qwen3_batch_size', value)
            value = int(value)
        d['qwen3_batch_size'] = value

        seed = d.get('qwen3_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('qwen3_seed', -1)
            seed = -1
        d['qwen3_seed'] = int(seed)

        from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
        value = d.get('pocket_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (PocketBaseModel.TEMPERATURE_MIN <= value <= PocketBaseModel.TEMPERATURE_MAX):
                value = -1
                add_warning('pocket_temperature', value)
        d['pocket_temperature'] = value

        seed = d.get('pocket_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('pocket_seed', -1)
            seed = -1
        d['pocket_seed'] = int(seed)

        d['pocket_model_code'] = d.get('pocket_model_code', '')

        seed = d.get("omnivoice_seed", -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning("omnivoice_seed", -1)
            seed = -1
        d["omnivoice_seed"] = int(seed)

        value = d.get("omnivoice_num_step", -1)
        if value != -1:
            if not isinstance(value, int) or not (OmniVoiceBaseModel.MIN_STEPS <= value <= OmniVoiceBaseModel.MAX_STEPS):
                value = -1
                add_warning("omnivoice_num_step", value)
        d["omnivoice_num_step"] = int(value)

        value = d.get("omnivoice_speed", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (0.5 <= value <= 2.0):
                value = -1
                add_warning("omnivoice_speed", value)
        d["omnivoice_speed"] = value

        value = d.get("omnivoice_cfg", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (OmniVoiceBaseModel.CFG_MIN <= value <= OmniVoiceBaseModel.CFG_MAX):
                value = -1
                add_warning("omnivoice_cfg", value)
        d["omnivoice_cfg"] = value

        return d

    @staticmethod
    def to_project_json_dict(project: Project) -> dict:
        return {
            "dir_path": project.dir_path,
            "version": project.version,

            "language_code": project.language_code,

            "segmentation_strategy": project.segmentation_strategy.id,
            "max_words": project.max_words,
            "word_substitutions_json_string": json.dumps(project.word_substitutions),

            "generate_range": project.generate_range_string,
            "markers": project.markers,
            "subdivide_phrases": project.subdivide_phrases,
            "export_type": project.export_type.id,
            "use_break_sound_effect": project.use_break_sound_effect,
            "normalization_type": project.normalization_type.value.id,
            "high_shelf": project.high_shelf,
            "use_upsampler": project.use_upsampler,
            "realtime_save": project.realtime_save,
            "realtime_line_range": project.realtime_line_range,
            "limit_silence_gaps": project.limit_silence_gaps,
            "limit_silence_gaps_duration": project.limit_silence_gaps_duration,
            "gen_auto_concat": project.gen_auto_concat,
            "streaming_chat": project.streaming_chat,
            "strictness": project.strictness.id,
            "max_retries": project.max_retries,
            "chapter_mode": project.chapter_mode.id,

            "none_voice_file_name": project.none_voice_file_name,

            "oute_voice_file_name": project.oute_voice_file_name,
            "oute_temperature": project.oute_temperature,

            "chatterbox_type": project.chatterbox_type.id,
            "chatterbox_voice_file_name": project.chatterbox_voice_file_name,
            "chatterbox_temperature": project.chatterbox_temperature,
            "chatterbox_top_p": project.chatterbox_top_p,
            "chatterbox_turbo_top_k": project.chatterbox_turbo_top_k,
            "chatterbox_ml_repetition_penalty": project.chatterbox_ml_repetition_penalty,
            "chatterbox_turbo_repetition_penalty": project.chatterbox_turbo_repetition_penalty,
            "chatterbox_cfg": project.chatterbox_cfg,
            "chatterbox_exaggeration": project.chatterbox_exaggeration,
            "chatterbox_seed": project.chatterbox_seed,

            "fish_s1_voice_file_name": project.fish_s1_voice_file_name,
            "fish_s1_voice_text": project.fish_s1_voice_transcript,
            "fish_s1_temperature": project.fish_s1_temperature,
            "fish_s1_top_p": project.fish_s1_top_p,
            "fish_s1_repetition_penalty": project.fish_s1_repetition_penalty,
            "fish_s1_seed": project.fish_s1_seed,
            "fish_s1_compile_enabled": project.fish_s1_compile_enabled,

            "fish_s2_voice_file_name": project.fish_s2_voice_file_name,
            "fish_s2_voice_text": project.fish_s2_voice_transcript,
            "fish_s2_rolling_cont": project.fish_s2_rolling_cont,
            "fish_s2_temperature": project.fish_s2_temperature,
            "fish_s2_top_p": project.fish_s2_top_p,
            "fish_s2_top_k": project.fish_s2_top_k,
            "fish_s2_seed": project.fish_s2_seed,
            "fish_s2_compile_enabled": project.fish_s2_compile_enabled,
            "fish_s2_server_voice_target": project.fish_s2_server_voice_target,
            "fish_s2_server_voice_transcript": project.fish_s2_server_voice_transcript,
            "fish_s2_server_concurrent_requests": project.fish_s2_server_concurrent_requests,

            "higgs_voice_file_name": project.higgs_voice_file_name,
            "higgs_voice_text": project.higgs_voice_transcript,
            "higgs_temperature": project.higgs_temperature,
            "higgs_top_k": project.higgs_top_k,
            "higgs_top_p": project.higgs_top_p,
            "higgs_seed": project.higgs_seed,

            "higgs_v3_voice_target": project.higgs_v3_voice_target,
            "higgs_v3_voice_transcript": project.higgs_v3_voice_transcript,
            "higgs_v3_temperature": project.higgs_v3_temperature,
            "higgs_v3_top_p": project.higgs_v3_top_p,
            "higgs_v3_top_k": project.higgs_v3_top_k,
            "higgs_v3_batch_size": project.higgs_v3_batch_size,
            "higgs_v3_seed": project.higgs_v3_seed,

            "vibevoice_voice_file_name": project.vibevoice_voice_file_name,
            "vibevoice_target": project.vibevoice_target,
            "vibevoice_lora_path": project.vibevoice_lora_target,
            "vibevoice_cfg": project.vibevoice_cfg,
            "vibevoice_steps": project.vibevoice_steps,
            "vibevoice_batch_size": project.vibevoice_batch_size,
            "vibevoice_seed": project.vibevoice_seed,

            "indextts2_voice_file_name": project.indextts2_voice_file_name,
            "indextts2_temperature": project.indextts2_temperature,
            "indextts2_emo_voice_file_name": project.indextts2_emo_voice_file_name,
            "indextts2_emo_vector": project.indextts2_emo_vector,
            "indextts2_emo_alpha": project.indextts2_emo_alpha,
            "indextts2_use_fp16": project.indextts2_use_fp16,
            "indextts2_top_p": project.indextts2_top_p,
            "indextts2_top_k": project.indextts2_top_k,
            "indextts2_seed": project.indextts2_seed,

            "glm_voice_file_name": project.glm_voice_file_name,
            "glm_voice_text": project.glm_voice_transcript,
            "glm_sr": project.glm_sr,
            "glm_seed": project.glm_seed,

            "mira_voice_file_name": project.mira_voice_file_name,
            "mira_temperature": project.mira_temperature,
            "mira_top_p": project.mira_top_p,
            "mira_top_k": project.mira_top_k,
            "mira_repetition_penalty": project.mira_repetition_penalty,
            "mira_batch_size": project.mira_batch_size,
            "mira_seed": project.mira_seed,

            "moss_voice_file_name": project.moss_voice_file_name,
            "moss_voice_text": project.moss_voice_transcript,
            "moss_target": project.moss_target,
            "moss_rolling_cont": project.moss_rolling_cont,
            "moss_delay_temperature": project.moss_delay_temperature,
            "moss_delay_top_p": project.moss_delay_top_p,
            "moss_delay_top_k": project.moss_delay_top_k,
            "moss_local_temperature": project.moss_local_temperature,
            "moss_local_top_p": project.moss_local_top_p,
            "moss_local_top_k": project.moss_local_top_k,
            "moss_batch_size": project.moss_batch_size,
            "moss_seed": project.moss_seed,

            "qwen3_target": project.qwen3_target,
            "qwen3_model_type": project.qwen3_model_type,
            "qwen3_voice_file_name": project.qwen3_voice_file_name,
            "qwen3_voice_text": project.qwen3_voice_transcript,
            "qwen3_rolling_cont": project.qwen3_rolling_cont,
            "qwen3_speaker_id": project.qwen3_speaker_id,
            "qwen3_instructions": project.qwen3_instructions,
            "qwen3_batch_size": project.qwen3_batch_size,
            "qwen3_temperature": project.qwen3_temperature,
            "qwen3_top_k": project.qwen3_top_k,
            "qwen3_top_p": project.qwen3_top_p,
            "qwen3_repetition_penalty": project.qwen3_repetition_penalty,
            "qwen3_seed": project.qwen3_seed,

            "pocket_voice_file_name": project.pocket_voice_file_name,
            "pocket_predefined_voice": project.pocket_predefined_voice,
            "pocket_model_code": project.pocket_model_code,
            "pocket_temperature": project.pocket_temperature,
            "pocket_seed": project.pocket_seed,

            "omnivoice_voice_file_name": project.omnivoice_voice_file_name,
            "omnivoice_voice_transcript": project.omnivoice_voice_transcript,
            "omnivoice_target": project.omnivoice_target,
            "omnivoice_instruct": project.omnivoice_instruct,
            "omnivoice_cfg": project.omnivoice_cfg,
            "omnivoice_speed": project.omnivoice_speed,
            "omnivoice_num_step": project.omnivoice_num_step,
            "omnivoice_seed": project.omnivoice_seed
        }

    @staticmethod
    def to_snapshot_dict(project: Project) -> dict:
        return ProjectSerializationUtil.to_project_json_dict(project)
