"""Pure v4 voice migration, with interaction confined to explicit UI callers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from tts_audiobook_tool.app_support import path_norm
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, PROJECT_SPEC_VERSION


@dataclass(frozen=True)
class VoiceReferenceSource:
    owner: tuple[str, str]
    model_ids: tuple[str, ...]
    references: list[dict[str, str]]

    @property
    def label(self) -> str:
        from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
        labels = []
        for model_id in self.model_ids:
            model = TtsModelType.get_by_id(model_id)
            labels.append(f"{model.value.ui['proper_name']} {model_id}" if model.id == model_id else model_id)
        return ' / '.join(labels) or self.owner[1]


class VoiceReferenceMigrationRequired(ValueError):
    def __init__(self, sources: list[VoiceReferenceSource]):
        self.sources = sources
        details = '; '.join(f"{s.label}: {', '.join(r['file_name'] for r in s.references)}" for s in sources)
        super().__init__(
            'Voice clone migration requires a model-list choice. Open/import this project '
            f'interactively to choose which list to keep. Sources: {details}'
        )


class VoiceReferenceMigrationCancelled(ValueError):
    pass


# Optional per-entry crop fields (see docs-dev/voice-sample-crop.md). Times
# are stored as canonical decimal strings so the entry dicts remain
# dict[str, str] throughout the serialization/compat stack.
VOICE_CROP_SUBDIR = 'crops'
CROP_FILE_NAME_FIELD = 'crop_file_name'
CROP_START_FIELD = 'crop_start'
CROP_END_FIELD = 'crop_end'
CROP_TRANSCRIPT_FIELD = 'crop_transcript'
CROP_FIELDS = (CROP_FILE_NAME_FIELD, CROP_START_FIELD, CROP_END_FIELD, CROP_TRANSCRIPT_FIELD)


def normalize_crop_file_name(value: Any) -> str | None:
    """Accept only a FLAC basename; the managed directory is not persisted."""
    if not isinstance(value, str):
        return None
    if (not value.endswith('.flac') or value == '.flac'
            or any(char in value for char in ('/', '\\', ':', '\x00'))):
        return None
    return value


def format_crop_seconds(value: float) -> str:
    """Canonical persisted form of a crop boundary in seconds."""
    return str(float(value))


def parse_crop_range(ref: dict) -> tuple[float, float] | None:
    """Parse a validated entry's crop range, or None when no crop is set.

    Only meaningful for entries that already passed through
    `normalize_voice_references`; malformed values yield None.
    """
    if normalize_crop_file_name(ref.get(CROP_FILE_NAME_FIELD)) is None:
        return None
    try:
        start = float(ref[CROP_START_FIELD])
        end = float(ref[CROP_END_FIELD])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        return None
    return start, end


def copy_crop_fields(ref: dict) -> dict:
    """Copy the crop subset of an already-normalized entry."""
    return {name: ref[name] for name in CROP_FIELDS if name in ref}


def _normalize_crop_fields(ref: dict, where: str, index: int) -> dict:
    """Drop malformed crop metadata with a warning; never infer a crop path.

    The pre-deployment sibling naming scheme is intentionally unsupported.
    Original sample files and transcripts are retained untouched.
    """
    from tts_audiobook_tool.l import L

    if not any(name in ref for name in CROP_FIELDS):
        return {}

    crop_file_name = normalize_crop_file_name(ref.get(CROP_FILE_NAME_FIELD))
    if crop_file_name is None:
        L.w(f'{where}[{index}]: trim requires an explicit <name>.flac basename; dropping trim')
        return {}

    transcript = ref.get(CROP_TRANSCRIPT_FIELD, '')
    if not isinstance(transcript, str):
        L.w(f'{where}[{index}]: trim transcript must be a string; dropping trim')
        return {}

    parsed: tuple[float, float] | None = None
    try:
        start_raw = ref.get(CROP_START_FIELD)
        end_raw = ref.get(CROP_END_FIELD)
        if start_raw is None or end_raw is None:
            raise ValueError('trim start and end must both be present')
        start = float(start_raw)  # accepts numbers or numeric strings
        end = float(end_raw)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('expected finite seconds with 0 <= start < end')
        parsed = start, end
    except (TypeError, ValueError, OverflowError) as e:
        L.w(f'{where}[{index}]: invalid trim range ({e}); dropping trim')
        return {}

    assert parsed is not None
    start, end = parsed
    return {
        CROP_FILE_NAME_FIELD: crop_file_name,
        CROP_START_FIELD: format_crop_seconds(start),
        CROP_END_FIELD: format_crop_seconds(end),
        CROP_TRANSCRIPT_FIELD: transcript,
    }


def normalize_voice_references(value: Any, where: str = 'voice_references') -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f'{where} must be an array')
    result = []
    for index, ref in enumerate(value):
        if (not isinstance(ref, dict) or not isinstance(ref.get('file_name'), str)
                or not isinstance(ref.get('transcript', ''), str)):
            raise ValueError(f'{where}[{index}]: expected file_name and optional transcript strings')
        file_name, _ = path_norm.normalize_stored_relative_path(ref['file_name'])
        if not file_name:
            # A stored name that cannot be a project-local file (e.g. a stray
            # absolute path) drops that entry rather than failing the project,
            # matching the old flat-field handling of unusable paths.
            from tts_audiobook_tool.l import L
            L.w(f'{where}[{index}].file_name is not a usable project-local path; dropping entry')
            continue
        entry = {'file_name': file_name, 'transcript': ref.get('transcript', '')}
        crop_fields = _normalize_crop_fields(ref, where, index)
        if crop_fields:
            entry.update(crop_fields)
        result.append(entry)
    return result


def _legacy_strings(value: Any, where: str) -> list[str]:
    """Tolerant reader for legacy flat fields.

    Old flat fields were permissively flattened on load (non-strings and empty
    strings dropped), so sloppy historical data must not fail the project now
    that this data feeds the v4 list. Structured top-level `voice_references`
    entries stay strict; only these pre-v3 spellings are forgiving.
    """
    from tts_audiobook_tool.l import L
    if isinstance(value, str):
        return [value] if value else []
    if value is None:
        return []
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item:
                    strings.append(item)
            else:
                L.w(f'{where}: ignoring non-string entry {item!r}')
        return strings
    L.w(f'{where}: expected a string or array of strings; ignoring {value!r}')
    return []


def _warn_unusable_historical(group: str, detail: str) -> None:
    """Historical groups never block loading; retain them for the warning path."""
    from tts_audiobook_tool.l import L
    L.w(f'model_settings.shared.{group}: {detail}; historical group left for non-destructive retention')


def _collect_sources(d: dict[str, Any]) -> list[VoiceReferenceSource]:
    from tts_audiobook_tool.project_support.model_settings import REGISTRY, ModelSettings
    from tts_audiobook_tool.project_support.model_settings_compat import SHARED_MEMBERS
    from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil

    settings = d.get('model_settings', {})
    if settings is None:
        settings = {}
    if isinstance(settings, ModelSettings):
        settings = settings.to_dict()
    if not isinstance(settings, dict):
        raise ValueError('model_settings must be an object')
    sources: dict[tuple[str, str], VoiceReferenceSource] = {}
    known_models = {id for id, _ in REGISTRY.bindings}
    for section in ('models', 'shared'):
        objects = settings.get(section, {})
        if not isinstance(objects, dict):
            raise ValueError(f'model_settings.{section} must be an object')
        for key, obj in objects.items():
            if not isinstance(obj, dict):
                raise ValueError(f'model_settings.{section}.{key} must be an object')
            if 'voice_references' not in obj:
                continue
            if section == 'models':
                if key not in known_models:
                    # Unknown whole objects are opaque here: their scoped voice
                    # data is neither a migration source nor a parse error, and
                    # it is dropped rather than preserved.
                    continue
                ids = (key,)
                refs = normalize_voice_references(obj['voice_references'], f'model_settings.models.{key}.voice_references')
            else:
                if key not in REGISTRY.members and key not in SHARED_MEMBERS:
                    continue
                historical = key not in REGISTRY.members
                expected = REGISTRY.members.get(key, SHARED_MEMBERS.get(key))
                recorded = obj.get('model_ids', [])
                if not isinstance(recorded, list) or not all(isinstance(id, str) for id in recorded):
                    if historical:
                        _warn_unusable_historical(key, 'model_ids must be an array of strings')
                        continue
                    raise ValueError(f'model_settings.shared.{key}.model_ids must be an array of strings')
                if expected is not None and (len(recorded) != len(expected) or set(recorded) != set(expected)):
                    if historical:
                        _warn_unusable_historical(key, f'expected members {list(expected)!r}')
                        continue
                    raise ValueError(f'model_settings.shared.{key}.model_ids: expected members {list(expected)!r}')
                ids = tuple(recorded)
                try:
                    refs = normalize_voice_references(obj['voice_references'], f'model_settings.shared.{key}.voice_references')
                except ValueError:
                    if historical:
                        _warn_unusable_historical(key, 'unusable voice_references')
                        continue
                    raise
                if key == 'moss':
                    # Historical MOSS references only fill private lists that
                    # were not explicitly supplied (including empty lists).
                    models = settings.get('models', {})
                    ids = tuple(id for id in ids if 'voice_references' not in models.get(id, {}))
                    if not ids:
                        refs = []
            owner = (section, key)
            # Keep empty sources here: explicit empty lists mask flat legacy data.
            sources[owner] = VoiceReferenceSource(owner, ids, refs)

    for attr, binding in REGISTRY.legacy.items():
        if binding.section != 'voice_references' or binding.name != 'file_name' or attr not in d:
            continue
        owner = ('shared', binding.group) if binding.group else ('models', binding.model_id)
        # v3 owner objects have whole-object precedence over corresponding flat fields.
        objects = settings.get(owner[0], {})
        if owner[1] in objects:
            continue
        # MOSS historical sharing is consumed once, before private fan-out.
        ids = SHARED_MEMBERS.get(binding.group, (binding.model_id,)) if binding.group else (binding.model_id,)
        if binding.group == 'moss':
            ids = tuple(id for id in ids if 'voice_references' not in settings.get('models', {}).get(id, {}))
            if not ids:
                continue
        voices = _legacy_strings(d[attr], attr)
        transcript_attr = attr.removesuffix('_file_name') + '_transcript'
        transcript_value = d.get(transcript_attr, '')
        if transcript_attr not in d:
            transcript_value = next((d[alias] for alias in ProjectSerializationUtil.VOICE_LIST_FIELD_ALIASES.get(transcript_attr, ()) if alias in d), '')
        transcripts = _legacy_strings(transcript_value, transcript_attr)
        refs = normalize_voice_references([
            {'file_name': voice, 'transcript': transcripts[i] if i < len(transcripts) else ''}
            for i, voice in enumerate(voices) if voice
        ], attr)
        sources[owner] = VoiceReferenceSource(owner, ids, refs)
    return [source for source in sources.values() if source.references]


def choose_voice_reference_source(
    sources: list[VoiceReferenceSource], dir_name: str | None = None,
) -> VoiceReferenceSource:
    from tts_audiobook_tool import ask
    from tts_audiobook_tool.util import printt
    subject = f'Project {dir_name}' if dir_name else 'This project'
    printt(
        f'🔔 {COL_ACCENT}Action required:\n'
        f'{COL_DEFAULT}{subject} contains voice clone lists for multiple models, but the app \n'
        'now uses one voice clone list per project, shared by all models. \n'
        'Choose which model’s voice clone list to keep:'
    )
    printt()
    for index, source in enumerate(sources, 1):
        printt(f"[{index}] {source.label}: {COL_DIM} {', '.join(ref['file_name'] for ref in source.references)}")
    printt('[0] Abort')
    printt()
    while True:
        answer = ask.ask_input('Choose a number: ')
        if answer in ('0', 'abort', 'skip', 'cancel'):
            raise VoiceReferenceMigrationCancelled('Voice clone migration cancelled; project was not changed.')
        if answer.isdecimal() and 1 <= int(answer) <= len(sources):
            return sources[int(answer) - 1]
        printt(f'Please choose a number from 1 to {len(sources)}, or 0 to abort.')
        printt()


def prepare_project_voice_references(
    d: dict[str, Any], *, prompt: bool = False, dir_name: str | None = None,
) -> dict[str, Any]:
    """Return a copy with one authoritative list; never save or mutate the input."""
    from tts_audiobook_tool.project_support.model_settings import REGISTRY, ModelSettings
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil

    # Preserve live Book/PhraseGroup identity on in-memory validation paths.
    # Only settings objects can be changed by remapping/voice cleanup here.
    d = dict(d)
    if 'model_settings' in d:
        d['model_settings'] = deepcopy(d['model_settings'])
    version = d.get('version')
    if isinstance(version, int) and version > PROJECT_SPEC_VERSION:
        raise ValueError(f'Unsupported project format version {version}; update the app before opening it.')
    ProjectLoadUtil.remap_legacy_keys(d)
    if 'voice_references' in d:
        refs = normalize_voice_references(d['voice_references'])
    else:
        sources = _collect_sources(d)
        if len(sources) > 1:
            if not prompt:
                raise VoiceReferenceMigrationRequired(sources)
            refs = choose_voice_reference_source(sources, dir_name).references
        else:
            refs = sources[0].references if sources else []
    d['voice_references'] = deepcopy(refs)
    # Remove scoped copies from every object, including unknown models/groups:
    # v4 has exactly one authoritative list, and unrecognized scoped voice data
    # is dropped, not preserved as a second voice source.
    settings = d.get('model_settings')
    if isinstance(settings, ModelSettings):
        settings = settings.to_dict()
        d['model_settings'] = settings
    if isinstance(settings, dict):
        from tts_audiobook_tool.project_support.model_settings_compat import SHARED_MEMBERS
        for section in ('models', 'shared'):
            objects = settings.get(section, {})
            if not isinstance(objects, dict):
                continue
            for key, obj in objects.items():
                if not isinstance(obj, dict):
                    continue
                if section == 'shared' and key in SHARED_MEMBERS and key not in REGISTRY.members:
                    # Retired historical groups keep their object so the
                    # existing reconciliation warning path can retain a
                    # malformed group non-destructively; a well-formed group
                    # is consumed by the historical fork instead.
                    continue
                obj.pop('voice_references', None)
    for attr, binding in REGISTRY.legacy.items():
        if binding.section == 'voice_references':
            d.pop(attr, None)
            for alias in ProjectSerializationUtil.VOICE_LIST_FIELD_ALIASES.get(attr, ()):
                d.pop(alias, None)
    d['version'] = PROJECT_SPEC_VERSION
    return d
