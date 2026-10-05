from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, NamedTuple

from pydantic.fields import FieldInfo

from tts_audiobook_tool.app_support import path_norm
from tts_audiobook_tool.constants import (
    APP_META_FLAC_FIELD,
    APP_META_MP4_MEAN,
    APP_META_MP4_TAG,
    PROJECT_TEXT_EPUB_FILE_NAME,
    PROJECT_TEXT_FILE_NAME,
    PROJECT_TEXT_RAW_FILE_NAME,
    PROJECT_VOICE_SUBDIR,
)
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.project_support.model_settings import REGISTRY

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class SourceFileMatch(NamedTuple):
    """
    A supporting project file located on disk.

    `relative_path` is the canonical project-local form to write it under,
    which is not necessarily the form it was stored or found in.
    """
    path: str
    relative_path: str


class ProjectTransferUtil:
    """
    Snapshot import/export and supporting-project-file transfer helpers.
    """

    PROJECT_SETTINGS_TRANSFER_SKIP = {
        'dir_path',
        'sound_segments',
    }

    PROJECT_SETTINGS_TRANSFER_EXTRA_FIELDS = {
        # Stored in the project text payload rather than directly in project.json.
        'book',
        # Serialized as word_substitutions_json_string for backwards compatibility.
        'word_substitutions',
    }

    # Settings that hold a machine-local path the user typed, or that the app
    # recorded on their behalf, as `(model ID, setting name)` references. The
    # same setting may equally hold a model repository id, so these are treated
    # as opaque user strings: nothing here rewrites them and a failed existence
    # check never clears them. `looks_foreign` only produces a warning that the
    # value cannot name a file on this machine, instead of letting the value
    # fail obscurely later.
    MACHINE_LOCAL_PATH_TARGETS = (
        ('dots_local', 'target'),
        ('moss_local', 'target'),
        ('omnivoice_local', 'target'),
        ('pocket_local', 'model_code'),
        ('qwen3tts_local', 'target'),
        ('vibevoice_local', 'target'),
        ('vibevoice_local', 'lora_target'),
    )

    @staticmethod
    def load_raw_abr_metadata_string(abr_path: str) -> str:
        suffix = os.path.splitext(abr_path)[1].lower()
        if suffix == '.flac':
            return AudioMetaUtil.get_flac_metadata_field(abr_path, APP_META_FLAC_FIELD)
        if suffix in ['.m4a', '.m4b']:
            string, _ = AudioMetaUtil.get_mp4_metadata_tag(abr_path, APP_META_MP4_MEAN, APP_META_MP4_TAG)
            return string
        return ""

    @staticmethod
    def make_project_from_snapshot(project_dir: str, project_snapshot: dict) -> Project:
        from tts_audiobook_tool.project import Project

        parse_dict = dict(project_snapshot)
        parse_dict['dir_path'] = project_dir
        return Project.model_validate(parse_dict)

    @staticmethod
    def validate_abr_snapshot(project_snapshot: dict, *, prompt_on_migration: bool = False) -> Project:
        """Validate an ABR snapshot before creating or selecting a destination.

        Pydantic ignores unknown top-level keys, so a nonempty but unrelated
        object would otherwise import as a successful, empty project. Keep
        legacy flat fields valid while requiring at least one actual setting.
        """
        from tts_audiobook_tool.project import Project
        from tts_audiobook_tool.project_support.model_settings_declarations import BUILTIN_LEGACY_FIELDS
        from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
        from tts_audiobook_tool.project_support.voice_reference_migration import prepare_project_voice_references

        version = project_snapshot.get('version')
        if version is not None and (type(version) is not int or version < 1):
            raise ValueError('ABR project snapshot has an invalid project version')
        known = set(Project.model_fields) | set(BUILTIN_LEGACY_FIELDS)
        known.update(field.alias for field in Project.model_fields.values() if field.alias)
        known.update(ProjectSerializationUtil.LEGACY_INPUT_ALIASES)
        for aliases in ProjectSerializationUtil.LEGACY_INPUT_ALIASES.values():
            known.update(aliases)
        known.update(('text', 'text_segments', 'chapter_indices', 'word_substitutions_json_string'))
        known.difference_update(('version', 'dir_path', 'source_dir_display', 'model_settings'))
        has_settings = any(key in project_snapshot for key in known)
        if 'model_settings' in project_snapshot:
            if not isinstance(project_snapshot['model_settings'], dict):
                raise ValueError('ABR project snapshot model_settings must be an object')
            # Recognize only the container shape here. Reconciliation must run
            # after preparation removes stale scoped clone data superseded by
            # an authoritative top-level list (including an explicit []).
            for section in ('models', 'shared'):
                objects = project_snapshot['model_settings'].get(section, {})
                if not isinstance(objects, dict):
                    raise ValueError(f'ABR project snapshot model_settings.{section} must be an object')
                has_settings |= bool(objects)
        if not has_settings:
            raise ValueError('ABR project snapshot contains no recognizable project settings')
        # Check the original payload before preparation inserts voice_references
        # and stamps a version, so unrelated or invalid snapshots cannot pass.
        project_snapshot = prepare_project_voice_references(project_snapshot, prompt=prompt_on_migration)
        if 'model_settings' in project_snapshot:
            # Preserve ABR's strict validation of remaining model settings, but
            # only after obsolete scoped clone lists have been removed.
            REGISTRY.reconcile(project_snapshot['model_settings'])
        # An empty dir_path prevents validation from creating the destination's
        # sound-segments directory. The destination is set only at commit time.
        return ProjectTransferUtil.make_project_from_snapshot('', project_snapshot)

    @staticmethod
    def apply_project_settings(dest_project: Project, source_project: Project) -> None:
        for field_name in ProjectTransferUtil.get_project_settings_transfer_field_names(source_project):
            if field_name == 'markers':
                dest_project.markers = source_project.markers
                continue
            setattr(dest_project, field_name, getattr(source_project, field_name))

    @staticmethod
    def get_project_settings_transfer_field_names(project: Project | type[Project]) -> list[str]:
        model_fields = project.model_fields if isinstance(project, type) else type(project).model_fields
        return [
            'markers' if field_name == 'marker_indices' else field_name
            for field_name in model_fields
            if field_name not in ProjectTransferUtil.PROJECT_SETTINGS_TRANSFER_SKIP
        ]

    @staticmethod
    def get_missing_project_settings_transfer_fields(project: type[Project]) -> list[str]:
        serialized_field_names = ProjectTransferUtil.get_project_json_serialized_field_names(project)
        return [
            field_name
            for field_name in ProjectTransferUtil.get_project_settings_transfer_field_names(project)
            if field_name not in serialized_field_names
            and field_name not in ProjectTransferUtil.PROJECT_SETTINGS_TRANSFER_EXTRA_FIELDS
        ]

    @staticmethod
    def get_project_json_serialized_field_names(project: type[Project]) -> set[str]:
        from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil

        payload = ProjectSerializationUtil.to_project_json_dict(project())
        alias_to_field_name = ProjectTransferUtil.make_project_alias_to_field_name_map(project)
        return {
            'markers' if key == 'markers' else alias_to_field_name[key] if key in alias_to_field_name else key
            for key in payload
        }

    @staticmethod
    def make_project_alias_to_field_name_map(project: type[Project]) -> dict[str, str]:
        result: dict[str, str] = {}
        for field_name, field_info in project.model_fields.items():
            if isinstance(field_info, FieldInfo) and field_info.alias:
                result[str(field_info.alias)] = field_name
        return result

    @staticmethod
    def find_foreign_path_targets(project: Project) -> list[tuple[str, str]]:
        """
        Machine-local path settings that cannot name a file on this machine.

        Returned for messaging only. The values are left untouched because the
        same fields legitimately hold repository ids, and because a value that
        is merely unresolvable right now may become resolvable later.
        """
        result: list[tuple[str, str]] = []
        for model_id, name in ProjectTransferUtil.MACHINE_LOCAL_PATH_TARGETS:
            value = project.get_model_setting(model_id, name)
            if isinstance(value, str) and value and path_norm.looks_foreign(value):
                result.append((f"{model_id}_{name}", value))
        return result

    @staticmethod
    def get_snapshot_source_dir_display(project_snapshot: dict) -> str:
        """
        The project directory the snapshot's settings came from.

        Display-only. Current snapshots store it under `source_dir_display`;
        older snapshots stored it as `dir_path`, which the app used to treat as
        resolvable. Both spellings are in circulation and the ABR version is
        unchanged either way: the difference is producer-side.
        """
        for key in ('source_dir_display', 'dir_path'):
            value = project_snapshot.get(key, '')
            if isinstance(value, str) and value:
                return value
        return ''

    @staticmethod
    def get_snapshot_source_dir(project_snapshot: dict, abr_path: str = '') -> str:
        """
        Locate a usable project directory for an ABR file's settings snapshot.

        The stored directory belongs to the creating machine's path grammar, so
        it is only used when it actually resolves here — the common case being
        an ABR file re-imported on the machine that produced it. Otherwise the
        ABR file's own location is the best available evidence: generated ABR
        files are written inside the project directory, so both the file's
        directory and its parent are tried.
        """
        candidates: list[str] = []

        stored_dir = ProjectTransferUtil.get_snapshot_source_dir_display(project_snapshot)
        if stored_dir:
            candidates.append(stored_dir)

        if abr_path:
            abr_dir = os.path.dirname(os.path.abspath(abr_path))
            candidates.extend([abr_dir, os.path.dirname(abr_dir)])

        for candidate in candidates:
            if ProjectTransferUtil.is_supporting_project_source_dir(candidate):
                return candidate

        return ''

    @staticmethod
    def is_supporting_project_source_dir(dir_path: str) -> bool:
        """Whether a directory looks like the project directory it came from."""
        if not dir_path or not os.path.isdir(dir_path):
            return False
        if os.path.exists(os.path.join(dir_path, PROJECT_TEXT_FILE_NAME)):
            return True
        return os.path.isdir(os.path.join(dir_path, PROJECT_VOICE_SUBDIR))

    @staticmethod
    def make_supporting_project_file_names(project: Project) -> list[str]:
        text_file_names, voice_file_names = ProjectTransferUtil.collect_supporting_project_file_names(project)
        return text_file_names + voice_file_names

    @staticmethod
    def collect_supporting_project_file_names(project: Project) -> tuple[list[str], list[str]]:
        """
        The project's supporting files, classified by where they belong in a
        destination project: `(text file names, voice sample file names)`.

        Text files live at the project root; voice sample files live in the
        project's voice subdir, wherever the source project happened to keep
        them.
        """
        raw_text_file_names: list[object] = [
            PROJECT_TEXT_FILE_NAME,
            PROJECT_TEXT_RAW_FILE_NAME,
            PROJECT_TEXT_EPUB_FILE_NAME,
        ]
        # The clone list is project-wide. Collect it once, then independently
        # collect catalog-declared secondary files such as IndexTTS emo_voice.
        raw_voice_file_names: list[object] = [entry.get("file_name") for entry in project.voice_references]
        seen_owners: set[tuple[str, str]] = set()
        for model_type in TtsModelType.all():
            for binding in REGISTRY.for_model(model_type.id):
                if binding.section != "files":
                    continue
                owner = (binding.group or binding.model_id, binding.name)
                if owner not in seen_owners:
                    seen_owners.add(owner)
                    raw_voice_file_names.append(project.get_model_setting(model_type.id, binding.name))

        return (
            ProjectTransferUtil.filter_project_file_names(raw_text_file_names),
            ProjectTransferUtil.filter_project_file_names(raw_voice_file_names),
        )

    @staticmethod
    def filter_project_file_names(raw_file_names: list[object]) -> list[str]:
        """
        Reduces raw collected file settings to a deduplicated list of
        project-local file names.
        """
        filtered_file_names: list[str] = []
        for raw_file_name in raw_file_names:
            if not isinstance(raw_file_name, str) or not raw_file_name:
                continue

            # Values written by an older build, or by a project object that
            # never passed through the load-time normalization funnel, may still
            # be absolute in some other machine's grammar. Reduce them to the
            # project-local form so they can be looked for here.
            file_name, _ = path_norm.normalize_stored_relative_path(raw_file_name)
            if not file_name:
                continue
            if file_name not in filtered_file_names:
                filtered_file_names.append(file_name)

        return filtered_file_names

    @staticmethod
    def copy_supporting_project_files(
        project: Project,
        source_dir: str,
        text_file_names: list[str],
        voice_file_names: list[str],
        *,
        strict_copy_errors: bool = False,
    ) -> list[str]:
        """
        Copies text files to the destination project root and voice sample
        files into its voice subdir — wherever in the source project the files
        happened to be found, including the legacy project-root layout.
        """
        if not isinstance(source_dir, str):
            source_dir = ''

        missing_paths: list[str] = []

        for file_names, is_voice_file in ((text_file_names, False), (voice_file_names, True)):
            for file_name in file_names:
                match = ProjectTransferUtil.find_supporting_project_file_source_path(source_dir, file_name)
                if not match.path:
                    missing_paths.append(
                        path_norm.join_project_relative(source_dir, file_name) if source_dir else file_name
                    )
                    continue

                if is_voice_file:
                    # Saved voice references are bare names, which resolve
                    # against the voice subdir first, so the copy's place in
                    # the destination is the voice subdir regardless of where
                    # the source project kept it.
                    base_name = path_norm.split_relative(match.relative_path)[-1]
                    relative_path = f"{PROJECT_VOICE_SUBDIR}/{base_name}"
                else:
                    relative_path = match.relative_path

                dest_path = path_norm.join_project_relative(project.dir_path, relative_path)
                try:
                    dest_dir = os.path.dirname(dest_path)
                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(match.path, dest_path)
                except Exception as exc:
                    if strict_copy_errors:
                        raise OSError(f"Could not copy supporting file {match.path} to {dest_path}: {exc}") from exc
                    missing_paths.append(match.path)

        return missing_paths

    @staticmethod
    def find_supporting_project_file_source_path(source_dir: str, file_name: str) -> SourceFileMatch:
        """
        Look for a project-local file under `source_dir`.

        Searches the source root and its voice subdir, in the canonical
        relative form and by bare file name, so a file laid out by either
        operating system is found. The returned `relative_path` reflects where
        the file was found; `copy_supporting_project_files` upgrades legacy
        root-level voice files into the destination's voice subdir —
        previously derived with `os.path.commonpath`, which raises when the
        two paths belong to different path grammars.
        """
        name_parts = path_norm.split_relative(file_name)
        if not name_parts:
            return SourceFileMatch("", "")

        base_name = name_parts[-1]

        candidate_names = ["/".join(name_parts)]
        if len(name_parts) > 1:
            candidate_names.append(base_name)
        if file_name == PROJECT_TEXT_RAW_FILE_NAME:
            candidate_names.append("text_raw.txt")

        candidate_dirs: list[tuple[str, str]] = []
        if source_dir:
            candidate_dirs.append((source_dir, ""))
            candidate_dirs.append((os.path.join(source_dir, PROJECT_VOICE_SUBDIR), PROJECT_VOICE_SUBDIR))
        else:
            candidate_dirs.append(("", ""))

        for candidate_dir, dir_prefix in candidate_dirs:
            for candidate_name in candidate_names:
                if not candidate_dir:
                    candidate_path = candidate_name
                else:
                    candidate_path = path_norm.join_project_relative(candidate_dir, candidate_name)

                if not os.path.exists(candidate_path):
                    continue

                if dir_prefix:
                    relative_path = f"{dir_prefix}/{base_name}"
                else:
                    relative_path = candidate_name
                return SourceFileMatch(candidate_path, relative_path)

        return SourceFileMatch("", "")
