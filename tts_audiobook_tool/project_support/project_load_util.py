from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import Book, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import BOOK_FORMAT, book_from_project_text_json_dict, get_project_text_format
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME, PROJECT_TEXT_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import printt

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


@dataclass
class ProjectTextLoadResult:
    book: Book
    format: str


class ProjectLoadUtil:
    """
    Project loading, validation, and legacy migration helpers.
    """

    @staticmethod
    def load_using_dir_path(dir_path: str) -> Project | str:
        from tts_audiobook_tool import ask
        from tts_audiobook_tool import project as project_module
        from tts_audiobook_tool.project import Project

        if not os.path.exists(dir_path):
            return f"Project directory doesn't exist:\n{dir_path}"

        project_dict_path = os.path.join(dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(project_dict_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            return f"Error loading project settings: {e}"

        if not isinstance(d, dict):
            return f"Project settings file bad type: {type(d)}"

        had_legacy_applied_fields = any(
            key in d for key in (
                "applied_language_code",
                "applied_strategy",
                "applied_max_words",
            )
        )

        d['dir_path'] = dir_path

        inline_text_source = ""
        external_text_source = ""
        if "text" in d:
            inline_text_source = "text"
        elif "text_segments" in d:
            inline_text_source = "text_segments"
        elif 'phrase_groups' not in d and 'book' not in d:
            result = ProjectLoadUtil.load_book_payload(dir_path, d)
            if isinstance(result, str):
                return result
            if result is not None:
                d['book'] = result.book
                external_text_source = result.format

        thread_local = project_module._tl
        thread_local.warnings = []
        try:
            project = Project.model_validate(d)
        except Exception as e:
            return f"Failed to parse project: {e}"

        project._phrase_groups_dirty = False
        project._phrase_groups_inline_source = inline_text_source

        if inline_text_source:
            err = project.save(force_phrase_groups=True)
            if err:
                return err
            L.i(
                f"Migrated inline project phrase groups from project.json[{inline_text_source!r}] "
                f"to {PROJECT_TEXT_FILE_NAME}: {dir_path}"
            )

        if external_text_source and external_text_source != BOOK_FORMAT:
            err = project.save(force_phrase_groups=True)
            if err:
                return err
            L.i(
                f"Migrated {PROJECT_TEXT_FILE_NAME} from {external_text_source!r} "
                f"to {BOOK_FORMAT}: {dir_path}"
            )

        if had_legacy_applied_fields and not inline_text_source and external_text_source == BOOK_FORMAT:
            err = project.save()
            if err:
                return err
            L.i(f"Removed legacy applied text fields from {PROJECT_JSON_FILE_NAME}: {dir_path}")

        from tts_audiobook_tool.tts import Tts

        if Tts.get_type() == TtsModelInfos.OUTE:
            ProjectVoiceUtil.load_oute_voice_json(project)

        did_clear_invalid_voice_files = ProjectVoiceUtil.verify_voice_files_exist(project)

        pending = getattr(thread_local, 'warnings', [])
        thread_local.warnings = []
        if pending or did_clear_invalid_voice_files:
            project.save()
            for warning in pending:
                printt(warning)
            ask.ask_enter_to_continue()

        project._autosave = True
        return project

    @staticmethod
    def is_valid_project_dir(project_dir: str) -> str:
        if not os.path.exists(project_dir):
            return f"Doesn't exist: {project_dir}"

        items = os.listdir(project_dir)
        if not items:
            return ""
        if PROJECT_JSON_FILE_NAME in items:
            return ""
        return f"{project_dir} does not appear to be a project directory"

    @staticmethod
    def remap_legacy_keys(d: dict) -> None:
        for old, new in [
            ('fish_voice_file_name', 'fish_s1_voice_file_name'),
            ('fish_voice_text', 'fish_s1_voice_text'),
            ('fish_temperature', 'fish_s1_temperature'),
            ('fish_seed', 'fish_s1_seed'),
            ('higgs_v3_voice_text', 'higgs_v3_voice_transcript'),
        ]:
            if new not in d and old in d:
                d[new] = d.pop(old)
        if 'vibevoice_target' not in d and 'vibevoice_model_path' in d:
            d['vibevoice_target'] = d.pop('vibevoice_model_path', '')
        if 'qwen3_target' not in d and 'qwen3_path_or_id' in d:
            d['qwen3_target'] = d.pop('qwen3_path_or_id', '')
        if d.get('indextts2_emo_alpha', -1) == -1:
            value = d.get('indextts2_emo_voice_alpha', -1)
            if isinstance(value, (int, float)) and value >= 0:
                d['indextts2_emo_alpha'] = value

    @staticmethod
    def load_book_payload(project_dir: str, project_settings: dict) -> ProjectTextLoadResult | str | None:
        file_path = os.path.join(project_dir, PROJECT_TEXT_FILE_NAME)
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
        except Exception as e:
            return f"Error loading project text: {e}"

        format_value = get_project_text_format(payload)
        if not format_value:
            return f"Unsupported project text format"

        strategy = SegmentationStrategy.from_id(project_settings.get('applied_strategy', ''))
        legacy_settings = BookSegmentationSettings(
            language_code=project_settings.get('applied_language_code', ''),
            max_words_per_segment=project_settings.get('applied_max_words', 0),
            strategy=strategy or BookSegmentationSettings().strategy,
        )
        result = book_from_project_text_json_dict(payload, legacy_settings)
        if isinstance(result, str):
            return f"Error parsing project text: {result}"
        return ProjectTextLoadResult(book=result, format=format_value)

    @staticmethod
    def load_phrase_groups_payload(dir_path: str) -> list[PhraseGroup] | str | None:
        file_path = os.path.join(dir_path, PROJECT_TEXT_FILE_NAME)
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
        except Exception as e:
            return f"Error loading project text: {e}"

        if isinstance(payload, dict):
            if 'phrase_groups' not in payload:
                return "Project text file missing 'phrase_groups'"
            phrase_group_dicts = payload['phrase_groups']
        elif isinstance(payload, list):
            phrase_group_dicts = payload
        else:
            return f"Project text file bad type: {type(payload)}"

        result = PhraseGroup.phrase_groups_from_json_list(phrase_group_dicts)
        if isinstance(result, str):
            return f"Error parsing project text: {result}"
        return result
