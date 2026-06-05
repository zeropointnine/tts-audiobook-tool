from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from pydantic.fields import FieldInfo

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class ProjectTransferUtil:
    """
    Snapshot import/export and supporting-project-file transfer helpers.
    """

    PROJECT_SETTINGS_TRANSFER_SKIP = {
        'dir_path',
        'sound_segments',
        # Runtime-loaded from oute_voice_file_name after the supporting JSON file is copied.
        'oute_voice_json',
    }

    PROJECT_SETTINGS_TRANSFER_EXTRA_FIELDS = {
        # Stored in the project text payload rather than directly in project.json.
        'book',
        # Serialized as word_substitutions_json_string for backwards compatibility.
        'word_substitutions',
        # Legacy compatibility fields used while loading older projects.
        'applied_strategy',
        'applied_max_words',
        'applied_language_code',
    }

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
    def apply_project_settings(dest_project: Project, source_project: Project) -> None:
        with dest_project.batch():
            for field_name in ProjectTransferUtil.get_project_settings_transfer_field_names(source_project):
                setattr(dest_project, field_name, getattr(source_project, field_name))

    @staticmethod
    def get_project_settings_transfer_field_names(project: Project | type[Project]) -> list[str]:
        model_fields = project.model_fields if isinstance(project, type) else type(project).model_fields
        return [
            field_name
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
        return {alias_to_field_name[key] if key in alias_to_field_name else key for key in payload}

    @staticmethod
    def make_project_alias_to_field_name_map(project: type[Project]) -> dict[str, str]:
        result: dict[str, str] = {}
        for field_name, field_info in project.model_fields.items():
            if isinstance(field_info, FieldInfo) and field_info.alias:
                result[str(field_info.alias)] = field_name
        return result

    @staticmethod
    def get_snapshot_source_dir(project_snapshot: dict) -> str:
        source_dir = project_snapshot.get('dir_path', '')
        if not isinstance(source_dir, str) or not source_dir:
            return ''
        return source_dir

    @staticmethod
    def make_supporting_project_file_names(project: Project) -> list[str]:

        file_names = [
            PROJECT_TEXT_FILE_NAME,
            PROJECT_TEXT_RAW_FILE_NAME,
            PROJECT_TEXT_EPUB_FILE_NAME,
        ]

        for model_info in TtsModelInfos:
            attrs = [model_info.value.voice_file_name_attr, *model_info.value.extra_file_attrs]
            for attr in attrs:
                if not attr:
                    continue
                file_names.append(getattr(project, attr, ""))

        filtered_file_names: list[str] = []
        for file_name in file_names:
            if not isinstance(file_name, str) or not file_name:
                continue
            if os.path.isabs(file_name):
                continue
            if file_name in filtered_file_names:
                continue
            filtered_file_names.append(file_name)

        return filtered_file_names

    @staticmethod
    def copy_supporting_project_files(project: Project, source_dir: str, file_names: list[str]) -> list[str]:
        if not isinstance(source_dir, str):
            source_dir = ''

        missing_paths: list[str] = []

        for file_name in file_names:
            src_path = ProjectTransferUtil.find_supporting_project_file_source_path(source_dir, file_name)
            if not src_path:
                missing_paths.append(os.path.join(source_dir, file_name) if source_dir else file_name)
                continue

            dest_path = os.path.join(project.dir_path, file_name)
            try:
                shutil.copy(src_path, dest_path)
            except Exception:
                missing_paths.append(src_path)

        return missing_paths

    @staticmethod
    def find_supporting_project_file_source_path(source_dir: str, file_name: str) -> str:
        from tts_audiobook_tool.constants import PROJECT_TEXT_RAW_FILE_NAME

        candidate_names = [file_name]
        if file_name == PROJECT_TEXT_RAW_FILE_NAME:
            candidate_names.append("text_raw.txt")

        for candidate_name in candidate_names:
            candidate_path = os.path.join(source_dir, candidate_name) if source_dir else candidate_name
            if os.path.exists(candidate_path):
                return candidate_path

        return ""
