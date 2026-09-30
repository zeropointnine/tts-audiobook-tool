from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from tts_audiobook_tool.app_support import path_norm
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
    VoiceSelectMode,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.project_support.model_settings import (
    REGISTRY,
    ModelSettings,
    normalize_paths,
)
from tts_audiobook_tool.project_support.model_settings_declarations import BUILTIN_LEGACY_FIELDS
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.dots_base_model import (
    DotsBaseModel,
    DotsCompileMode,
)
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.mira_base_model import MiraBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.util import printt

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


# Bounds for legacy (version-2) server-only fields, retained for migrating
# older projects after the legacy server classes were removed. They match the
# shipped JSON definitions for these models.
AUK_SPEED_DEFAULT = 1.0
AUK_SPEED_MIN = 0.5
AUK_SPEED_MAX = 2.0
ZONOS2_TOP_K_MIN = 1
ZONOS2_TOP_K_MAX = 200
ZONOS2_TEMPERATURE_MIN = 0.05
ZONOS2_TEMPERATURE_MAX = 2.0
ZONOS2_REPETITION_PENALTY_MIN = 1.0
ZONOS2_REPETITION_PENALTY_MAX = 2.0


class ProjectSerializationUtil:
    VOICE_LIST_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "auk_voice_file_name": (),
        "auk_voice_transcript": (),
        "chatterbox_voice_file_name": (),
        "dots_voice_file_name": (),
        "dots_voice_transcript": (),
        "fish_s1_voice_file_name": (),
        "fish_s1_voice_transcript": ("fish_s1_voice_text",),
        "fish_s2_voice_file_name": (),
        "fish_s2_voice_transcript": (),
        "glm_voice_file_name": (),
        "glm_voice_transcript": ("glm_voice_text",),
        "higgs_voice_file_name": (),
        "higgs_voice_transcript": ("higgs_voice_text",),
        "higgs_v3_voice_file_name": (),
        "higgs_v3_voice_transcript": (),
        "indextts2_voice_file_name": (),
        "mira_voice_file_name": (),
        "moss_voice_file_name": (),
        "moss_voice_transcript": (),
        "omnivoice_voice_file_name": (),
        "omnivoice_voice_transcript": (),
        "pocket_voice_file_name": (),
        "qwen3_voice_file_name": (),
        "qwen3_voice_transcript": (),
        "vibevoice_voice_file_name": (),
        "zonos2_server_voice_file_name": (),
    }

    @staticmethod
    def normalize_voice_list_value(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item]
        return []

    @staticmethod
    def serialize_voice_list_value(value: Any) -> str | list[str]:
        items = ProjectSerializationUtil.normalize_voice_list_value(value)
        # Backward compatibility: keep zero/one configured voice clone sample serialized
        # with the legacy string shape, and only use a JSON list when multiple samples exist.
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        return items

    # Project-local path fields that are not list-valued and therefore not in
    # `VOICE_LIST_FIELD_ALIASES`.
    SCALAR_PATH_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "none_voice_file_name": (),
        "indextts2_emo_voice_file_name": (),
    }

    # Older file spellings of flat model fields. Used only to detect that a
    # canonical field was supplied by an old project (the loops above and
    # `remap_legacy_keys` fold the value into the canonical key); the canonical
    # name owns the migrated value.
    LEGACY_INPUT_ALIASES: dict[str, tuple[str, ...]] = {
        "fish_s1_voice_transcript": ("fish_s1_voice_text",),
        "glm_voice_transcript": ("glm_voice_text",),
        "higgs_voice_transcript": ("higgs_voice_text",),
        "higgs_v3_voice_transcript": ("higgs_v3_voice_text",),
        "vibevoice_lora_target": ("vibevoice_lora_path",),
        "omnivoice_num_step": ("omnivoice_steps",),
        "moss_delay_temperature": ("moss_temperature",),
        "moss_delay_top_p": ("moss_top_p",),
        "moss_delay_top_k": ("moss_top_k",),
    }

    @classmethod
    def get_project_local_path_field_names(cls) -> list[str]:
        """
        Field names whose stored values are project-local paths.

        These are the only path values that have a portable canonical form.
        Every other path-bearing field is either machine-local (`dir_path`) or
        an opaque user string that may be a repository id (`*_target`).
        """
        names = [
            name
            for name in cls.VOICE_LIST_FIELD_ALIASES
            if name.endswith("_voice_file_name")
        ]
        names.extend(cls.SCALAR_PATH_FIELD_ALIASES)
        return names

    @staticmethod
    def normalize_stored_path_values(values: list[str]) -> tuple[list[str], bool]:
        """
        Canonicalize a list of stored project-local paths.

        Rewrites legacy values — the other OS's separators, and absolute paths
        that older builds stored in portable fields — into the canonical
        `/`-separated project-local form. Values that canonicalize to nothing
        are dropped. Returns the rewritten list and whether anything changed.
        """
        normalized: list[str] = []
        changed = False
        for value in values:
            canonical, value_changed = path_norm.normalize_stored_relative_path(value)
            changed = changed or value_changed
            if canonical:
                normalized.append(canonical)
        return normalized, changed

    @staticmethod
    def normalize_loaded_project_dict(d: Any, *, warnings: list[str] | None = None) -> Any:

        if not isinstance(d, dict):
            return d

        # Version 3 projects carry model settings in `model_settings`; the flat
        # model fields are then legacy remnants, not expected-but-missing
        # properties, so their absence must not warn.
        has_model_settings = isinstance(d.get("model_settings"), dict)

        def add_warning(attr_name: str, defaulting_to: Any) -> None:
            if has_model_settings and attr_name in BUILTIN_LEGACY_FIELDS:
                return
            s = f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Missing or invalid value for {COL_ACCENT}{attr_name}{COL_DEFAULT}\n"
            s += "This can occur if a new project property or feature has been\n"
            s += "added to the app since the last time you opened this project.\n"
            s += f"Setting to default: {defaulting_to}"
            s += "\n"
            if warnings is not None:
                warnings.append(s)

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
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, valid_types)
                or (isinstance(raw_value, float) and not math.isfinite(raw_value))
            ):
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

        # Version-3 migration: remember which flat model fields this project
        # actually supplied, after the legacy key remap above (an old project
        # may only carry a pre-rename key) but before the per-field loops below
        # fill every field with its default.
        legacy_supplied_keys = set(d) & set(BUILTIN_LEGACY_FIELDS)
        for attr, aliases in ProjectSerializationUtil.LEGACY_INPUT_ALIASES.items():
            if attr not in legacy_supplied_keys and any(alias in d for alias in aliases):
                legacy_supplied_keys.add(attr)

        path_field_changes: list[str] = []

        for key, aliases in ProjectSerializationUtil.VOICE_LIST_FIELD_ALIASES.items():
            raw_value = d.get(key, None)
            for alias in aliases:
                if raw_value is None and alias in d:
                    raw_value = d[alias]
            values = ProjectSerializationUtil.normalize_voice_list_value(raw_value)
            if key.endswith("_voice_file_name"):
                values, path_changed = ProjectSerializationUtil.normalize_stored_path_values(values)
                if path_changed:
                    path_field_changes.append(key)
            d[key] = values
            for alias in aliases:
                if alias in d:
                    d[alias] = d[key]

        for key, aliases in ProjectSerializationUtil.SCALAR_PATH_FIELD_ALIASES.items():
            for field in (key, *aliases):
                value = d.get(field, None)
                if not isinstance(value, str) or not value:
                    continue
                canonical, path_changed = path_norm.normalize_stored_relative_path(value)
                if path_changed:
                    d[field] = canonical
                    path_field_changes.append(field)

        if path_field_changes and warnings is not None:
            s = f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Rewrote project-local path(s) in this project's "
            s += "saved settings to the app's portable form.\n"
            s += "Stored paths are written by one operating system and read by another, so a\n"
            s += "path saved on a different OS cannot resolve here.\n"
            s += f"Updated field(s): {', '.join(sorted(set(path_field_changes)))}"
            s += "\n"
            warnings.append(s)

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

        normalize_bool('dialog_segmentation', False)

        s = d.get('generate_range', '')
        if isinstance(s, str):
            normalized_range = s.strip().lower()
            if normalized_range in ('all', 'a'):
                d['generate_range'] = ''
            elif normalized_range == 'none':
                d['generate_range'] = 'none'

        if 'markers' in d:
            phrase_groups = d.get('phrase_groups', [])
            if not phrase_groups and isinstance(d.get('book'), Book):
                phrase_groups = d['book'].phrase_groups
            lst = d['markers']
            if isinstance(lst, list):
                is_valid = all(
                    isinstance(idx, int) and not isinstance(idx, bool)
                    and idx < len(phrase_groups)
                    for idx in lst
                )
                if not is_valid:
                    printt(f"File cut points invalid: {lst}")
                    d['markers'] = []
                else:
                    d['markers'] = [idx for idx in lst if idx > 0]
            else:
                d['markers'] = []

        if 'markers' in d:
            d['marker_indices'] = d.pop('markers')

        raw_legacy_language = d.pop('applied_language_code', '')
        legacy_language = raw_legacy_language if isinstance(raw_legacy_language, str) else ''

        raw_legacy_max_words = d.pop('applied_max_words', 0)
        legacy_max_words = (
            raw_legacy_max_words
            if isinstance(raw_legacy_max_words, int)
            and not isinstance(raw_legacy_max_words, bool)
            and raw_legacy_max_words >= 0
            else 0
        )

        raw_legacy_strategy = d.pop('applied_strategy', '')
        legacy_strategy = (
            raw_legacy_strategy
            if isinstance(raw_legacy_strategy, SegmentationStrategy)
            else SegmentationStrategy.from_id(raw_legacy_strategy)
            if isinstance(raw_legacy_strategy, str)
            else None
        )

        raw_legacy_dialog_segmentation = d.pop('applied_dialog_segmentation', False)
        legacy_dialog_segmentation = (
            raw_legacy_dialog_segmentation
            if isinstance(raw_legacy_dialog_segmentation, bool)
            else False
        )

        legacy_settings = BookSegmentationSettings(
            language_code=legacy_language,
            max_words_per_segment=legacy_max_words,
            strategy=legacy_strategy or BookSegmentationSettings().strategy,
            dialog_segmentation=legacy_dialog_segmentation,
        )
        if not isinstance(d.get('book'), Book) and d.get('phrase_groups'):
            d['book'] = Book(
                sections=[BookSection(phrase_groups=d.get('phrase_groups', []))],
                segmentation_settings=legacy_settings,
                text_source_kind="legacy_flat",
                audio_source_kind="unknown",
            )

        normalize_by_id('export_type', ExportType.get_by_id, list(ExportType)[0])
        normalize_by_id('normalization_type', NormalizationType.from_id, list(NormalizationType)[0])

        reason_pause_type = ReasonPauseTypes.get_by_id(d.get('reason_pauses', ''))
        if reason_pause_type is None:
            reason_pause_type = ReasonPauseTypes.default()
        d['reason_pauses'] = reason_pause_type.value

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

        normalize_by_id('voice_select_mode', VoiceSelectMode.get_by_id, VoiceSelectMode.AUTO_ADVANCE)

        s = d.get('chatterbox_type', '')
        chatterbox_type = ChatterboxType.get_by_id(s)
        if not chatterbox_type:
            chatterbox_type = list(ChatterboxType)[0]
            if d.get('tts_model_type') == "chatterbox_local":
                add_warning('chatterbox_type', chatterbox_type.id)
        d['chatterbox_type'] = chatterbox_type

        seed = d.get('chatterbox_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('chatterbox_seed', -1)
            seed = -1
        d['chatterbox_seed'] = seed

        target = d.get("dots_target", "")
        if not isinstance(target, str) or (
            target and target not in DotsBaseModel.PRESET_REPO_IDS
        ):
            target = ""
            add_warning("dots_target", target)
        d["dots_target"] = target

        seed = d.get("dots_seed", DotsBaseModel.SEED_DEFAULT)
        if not isinstance(seed, (int, float)) or not (
            DotsBaseModel.SEED_MIN <= seed <= DotsBaseModel.SEED_MAX
        ):
            seed = DotsBaseModel.SEED_DEFAULT
            add_warning("dots_seed", seed)
        d["dots_seed"] = int(seed)

        value = d.get("dots_speaker_scale", -1)
        if value != -1 and (
            not isinstance(value, (int, float))
            or not DotsBaseModel.SPEAKER_SCALE_MIN
            <= value
            <= DotsBaseModel.SPEAKER_SCALE_MAX
        ):
            value = -1
            add_warning("dots_speaker_scale", value)
        d["dots_speaker_scale"] = float(value)

        value = d.get("dots_num_steps_soar", -1)
        if value != -1 and (
            not isinstance(value, (int, float))
            or not DotsBaseModel.NUM_STEPS_SOAR_MIN
            <= value
            <= DotsBaseModel.NUM_STEPS_SOAR_MAX
        ):
            value = -1
            add_warning("dots_num_steps_soar", value)
        d["dots_num_steps_soar"] = int(value)

        value = d.get("dots_num_steps_mf", -1)
        if value != -1 and (
            not isinstance(value, (int, float))
            or not DotsBaseModel.NUM_STEPS_MF_MIN
            <= value
            <= DotsBaseModel.NUM_STEPS_MF_MAX
        ):
            value = -1
            add_warning("dots_num_steps_mf", value)
        d["dots_num_steps_mf"] = int(value)

        value = d.get("dots_guidance_scale", -1)
        if value != -1 and (
            not isinstance(value, (int, float))
            or not DotsBaseModel.GUIDANCE_SCALE_MIN
            <= value
            <= DotsBaseModel.GUIDANCE_SCALE_MAX
        ):
            value = -1
            add_warning("dots_guidance_scale", value)
        d["dots_guidance_scale"] = float(value)

        # A missing "dots_compile" is expected for projects predating the setting;
        # only warn when the key exists but holds an invalid value.
        d.setdefault("dots_compile", DotsCompileMode.default().enabled)
        normalize_bool("dots_compile", DotsCompileMode.default().enabled, warn=True)

        normalize_bool('fish_s1_compile_enabled', True)

        seed = d.get('fish_s1_seed', -1)
        if not (-1 <= seed <= SEED_MAX):
            add_warning('fish_s1_seed', -1)
            seed = -1
        d['fish_s1_seed'] = seed

        normalize_int('auk_server_concurrent_requests', 1, min_value=1, max_value=PROJECT_CONCURRENT_REQUESTS_MAX, warn=True)

        value = d.get('auk_speed', AUK_SPEED_DEFAULT)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not AUK_SPEED_MIN
            <= value
            <= AUK_SPEED_MAX
        ):
            value = AUK_SPEED_DEFAULT
            add_warning('auk_speed', value)
        d['auk_speed'] = float(value)

        normalize_int('auk_seed', -1, min_value=-1, max_value=SEED_MAX, warn=True)

        normalize_bool('fish_s2_compile_enabled', True)
        normalize_int('fish_s2_rolling_cont', 0, min_value=0, max_value=3, warn=True)
        normalize_int('fish_s2_server_concurrent_requests', 1, min_value=1, max_value=PROJECT_BATCH_SIZE_MAX, warn=True)
        normalize_int('qwen3_server_concurrent_requests', 1, min_value=1, max_value=PROJECT_BATCH_SIZE_MAX, warn=True)
        normalize_int('zonos2_server_concurrent_requests', 1, min_value=1, max_value=PROJECT_BATCH_SIZE_MAX, warn=True)

        value = d.get('zonos2_top_k', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (ZONOS2_TOP_K_MIN <= value <= ZONOS2_TOP_K_MAX):
                value = -1
                add_warning('zonos2_top_k', value)
            value = int(value)
        d['zonos2_top_k'] = value

        value = d.get('zonos2_temperature', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (ZONOS2_TEMPERATURE_MIN <= value <= ZONOS2_TEMPERATURE_MAX):
                value = -1
                add_warning('zonos2_temperature', value)
        d['zonos2_temperature'] = value

        value = d.get('zonos2_repetition_penalty', -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (ZONOS2_REPETITION_PENALTY_MIN <= value <= ZONOS2_REPETITION_PENALTY_MAX):
                value = -1
                add_warning('zonos2_repetition_penalty', value)
        d['zonos2_repetition_penalty'] = value

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

        ProjectSerializationUtil._migrate_model_settings_objects(d, legacy_supplied_keys, warnings=warnings)

        return d

    @staticmethod
    def _migrate_model_settings_objects(
        d: dict, legacy_supplied_keys: set[str], *, warnings: list[str] | None = None,
    ) -> None:
        """
        Convert validated flat model fields plus any supplied model-settings
        objects into one reconciled store, then remove the flat fields.

        New objects win over flat values; absent overrides are simply absent,
        never a persisted default sentinel. Whole unknown model/group objects
        are retained for later environments.
        """
        legacy_values: dict[str, Any] = {}
        for attr in legacy_supplied_keys:
            value = d.get(attr)
            if value is None:
                aliases = ProjectSerializationUtil.LEGACY_INPUT_ALIASES.get(attr, ())
                value = next((d[alias] for alias in aliases if alias in d), None)
            if value is not None:
                legacy_values[attr] = value

        try:
            settings = REGISTRY.reconcile(d.get("model_settings"), legacy=legacy_values or None)
        except ValueError as exc:
            # Actionable but non-destructive: keep whatever whole objects parsed
            # so the project still loads, and surface the reason.
            settings = ModelSettings()
            source = d.get("model_settings")
            if isinstance(source, dict):
                if isinstance(source.get("models"), dict):
                    settings.models.update(source["models"])
                if isinstance(source.get("shared"), dict):
                    settings.shared.update(source["shared"])
            if warnings is not None:
                warnings.append(f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Problem reading model settings objects:\n{exc}\n")

        if normalize_paths(settings) and warnings is not None:
            s = f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Rewrote project-local path(s) in this project's "
            s += "saved model settings to the app's portable form."
            s += "\n"
            warnings.append(s)

        d["model_settings"] = settings.to_dict()

        removal_keys: set[str] = set(BUILTIN_LEGACY_FIELDS)
        for aliases in ProjectSerializationUtil.LEGACY_INPUT_ALIASES.values():
            removal_keys.update(aliases)
        for key in removal_keys:
            d.pop(key, None)

    @staticmethod
    def canonicalize_project_local_paths(d: dict) -> None:
        """
        Write project-local path fields in the portable canonical form.

        Applied to the finished dict rather than at each emission site, so a
        newly added voice field is covered automatically.
        """
        for key in ProjectSerializationUtil.get_project_local_path_field_names():
            value = d.get(key, None)
            if isinstance(value, list):
                d[key], _ = ProjectSerializationUtil.normalize_stored_path_values(
                    [item for item in value if isinstance(item, str)]
                )
            elif isinstance(value, str) and value:
                d[key], _ = path_norm.normalize_stored_relative_path(value)

    @staticmethod
    def to_project_json_dict(project: Project) -> dict:
        result: dict[str, Any] = {
            # Machine-local record, kept for older builds and external tooling.
            # Never trusted when read back: both load paths overwrite it with
            # the directory actually being opened.
            "dir_path": project.dir_path,
            "version": project.version,
            "tts_model_type": project.tts_model_type,

            "language_code": project.language_code,

            "segmentation_strategy": project.segmentation_strategy.id,
            "max_words": project.max_words,
            "dialog_segmentation": project.dialog_segmentation,
            "word_substitutions_json_string": json.dumps(project.word_substitutions),

            "generate_range": project.generate_range_string,
            "markers": sorted(project.markers),
            "subdivide_phrases": project.subdivide_phrases,
            "export_type": project.export_type.id,
            "use_break_sound_effect": project.use_break_sound_effect,
            "normalization_type": project.normalization_type.value.id,
            "high_shelf": project.high_shelf,
            "reason_pauses": project.reason_pauses.id,
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
            "voice_select_mode": project.voice_select_mode.id,

            "none_voice_file_name": project.none_voice_file_name,

            "model_settings": ProjectSerializationUtil._serialize_model_settings(project),
        }

        ProjectSerializationUtil.canonicalize_project_local_paths(result)
        return result

    @staticmethod
    def _serialize_model_settings(project: Project) -> dict:
        # Save-time canonicalization of declared project-local paths, matching
        # the old flat-field behavior for values that never passed through the
        # load funnel.
        normalize_paths(project.model_settings)
        return REGISTRY.serialize(project.model_settings)

    @staticmethod
    def to_snapshot_dict(project: Project) -> dict:
        """
        The settings snapshot embedded in an ABR audio file's metadata.

        `dir_path` is deliberately excluded: it is an absolute path in the
        writing machine's grammar, and an ABR file is meant to be shared.
        `source_dir_display` carries the same value for messaging only, so a
        recipient can be told where the settings came from without the app
        ever treating it as a resolvable location.
        """
        snapshot = ProjectSerializationUtil.to_project_json_dict(project)
        snapshot.pop("dir_path", None)
        snapshot["source_dir_display"] = project.dir_path
        return snapshot
