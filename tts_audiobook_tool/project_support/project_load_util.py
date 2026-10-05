from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import Book, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import BOOK_FORMAT, book_from_project_text_json_dict, get_project_text_format
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, PROJECT_JSON_FILE_NAME, PROJECT_TEXT_FILE_NAME, PROJECT_SPEC_VERSION
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
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
    def load_using_dir_path(
        dir_path: str,
        *,
        prompt_on_warnings: bool = True,
        prompt_on_migration: bool | None = None,
    ) -> Project | str:
        from tts_audiobook_tool import ask
        from tts_audiobook_tool.project import Project

        if not os.path.exists(dir_path):
            return f"Project directory doesn't exist:\n{dir_path}"

        project_dict_path = os.path.join(dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(project_dict_path, 'rb') as f:
                original_settings = f.read()
                d = json.loads(original_settings)
        except Exception as e:
            return f"Error loading project settings: {e}"

        if not isinstance(d, dict):
            return f"Project settings file bad type: {type(d)}"

        from tts_audiobook_tool.project_support.voice_reference_migration import (
            prepare_project_voice_references,
            VoiceReferenceMigrationRequired,
        )
        from tts_audiobook_tool.project_support.model_settings import REGISTRY
        from tts_audiobook_tool.project_support.model_settings_compat import SHARED_MEMBERS
        settings = d.get('model_settings', {})
        # Scoped voice lists that the v4 migration actually removes. Retired
        # historical shared groups are retained verbatim by both the migration
        # and the serializer, so counting them here would flag needs_v4_save
        # on every open and create a new numbered backup each time.
        had_scoped_voices = isinstance(settings, dict) and any(
            isinstance(objects, dict) and any(
                isinstance(obj, dict) and 'voice_references' in obj
                for key, obj in objects.items()
                if section != 'shared' or key not in SHARED_MEMBERS or key in REGISTRY.members
            )
            for section, objects in (('models', settings.get('models', {})), ('shared', settings.get('shared', {})))
        )
        needs_v4_save = (
            d.get('version') != PROJECT_SPEC_VERSION or 'voice_references' not in d or had_scoped_voices
            or any(attr in d for attr, binding in REGISTRY.legacy.items() if binding.section == 'voice_references')
        )
        try:
            d = prepare_project_voice_references(
                d, prompt=prompt_on_warnings if prompt_on_migration is None else prompt_on_migration,
                dir_name=os.path.basename(os.path.normpath(dir_path)) or dir_path,
            )
        except VoiceReferenceMigrationRequired:
            # Surfaced as a distinct exception so noninteractive callers (e.g.
            # the server) can fail with targeted remediation steps.
            raise
        except ValueError as exc:
            return str(exc)

        had_legacy_applied_fields = any(
            key in d for key in (
                "applied_language_code",
                "applied_strategy",
                "applied_max_words",
                "applied_dialog_segmentation",
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

        pending_warnings: list[str] = []
        try:
            project = Project.model_validate(d, context={'warnings': pending_warnings})
        except Exception as e:
            return f"Failed to parse project: {e}"

        if needs_v4_save:
            # No load-time migration may overwrite settings before their exact
            # original bytes are backed up. Never clobber an earlier backup.
            backup_path = project_dict_path + '.pre-v4.bak'
            suffix = 1
            while True:
                try:
                    with open(backup_path, 'xb') as f:
                        f.write(original_settings)
                        f.flush()
                        os.fsync(f.fileno())
                    break
                except FileExistsError:
                    suffix += 1
                    backup_path = project_dict_path + f'.pre-v4.bak.{suffix}'
                except OSError as exc:
                    return f'Could not back up project settings before v4 migration: {exc}'

        if inline_text_source:
            err = ProjectTextIOUtil.save_book(project)
            if not err:
                err = project.save()
            if err:
                return err
            L.i(
                f"Migrated inline project phrase groups from project.json[{inline_text_source!r}] "
                f"to {PROJECT_TEXT_FILE_NAME}: {dir_path}"
            )

        if external_text_source and external_text_source != BOOK_FORMAT:
            err = ProjectTextIOUtil.save_book(project)
            if not err:
                err = project.save()
            if err:
                return err
            L.i(
                f"Migrated {PROJECT_TEXT_FILE_NAME} from {external_text_source!r} "
                f"to {BOOK_FORMAT}: {dir_path}"
            )

        if needs_v4_save and not inline_text_source and not (external_text_source and external_text_source != BOOK_FORMAT):
            err = project.save()
            if err:
                return err

        if had_legacy_applied_fields and not inline_text_source and external_text_source == BOOK_FORMAT:
            err = project.save()
            if err:
                return err
            L.i(f"Removed legacy applied text fields from {PROJECT_JSON_FILE_NAME}: {dir_path}")

        # Persist path canonicalization and defaulted values.
        if pending_warnings:
            err = project.save()
            if err:
                return err

        # Voice sample files are deliberately not verified at load time:
        # missing references are kept for later, and problems surface through
        # readiness blockers and the pre-feature validate_voices flow instead
        # of a load-time interruption.

        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        foreign_targets = ProjectTransferUtil.find_foreign_path_targets(project)
        if foreign_targets:
            s = f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Some saved settings are paths that cannot"
            s += "name a file on this computer, so they were probably saved on a different one.\n"
            s += "They have been left as saved, but the model that uses them will not find them.\n"
            for attr_name, value in foreign_targets:
                s += f"- {COL_ACCENT}{attr_name}{COL_DEFAULT}: {value}\n"
            s += "Re-enter the path in this menu's model settings to fix it."
            pending_warnings.append(s)

        if pending_warnings:
            for warning in pending_warnings:
                printt(warning)
            if prompt_on_warnings:
                ask.ask_enter_to_continue()

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
        # The audio.cpp family includes dedicated English weights as well as
        # multilingual checkpoints; its former ID misleadingly named only one route.
        old_model_id = "chatterbox_multilingual_audiocpp"
        new_model_id = "chatterbox_audiocpp"
        if d.get("tts_model_type") == old_model_id:
            d["tts_model_type"] = new_model_id
        settings = d.get("model_settings")
        models = settings.get("models") if isinstance(settings, dict) else None
        if isinstance(models, dict) and old_model_id in models:
            old_settings = models.pop(old_model_id)
            current_settings = models.get(new_model_id)
            if isinstance(old_settings, dict) and isinstance(current_settings, dict):
                # Explicit settings under the new ID win, but retain old overrides
                # in sections/parameters not yet supplied under the new ID.
                for section, values in old_settings.items():
                    if section not in current_settings:
                        current_settings[section] = values
                    elif isinstance(values, dict) and isinstance(current_settings[section], dict):
                        current_settings[section] = {**values, **current_settings[section]}
            else:
                models.setdefault(new_model_id, old_settings)

        for old, new in [
            ('fish_voice_file_name', 'fish_s1_voice_file_name'),
            ('fish_voice_text', 'fish_s1_voice_text'),
            ('fish_temperature', 'fish_s1_temperature'),
            ('fish_seed', 'fish_s1_seed'),
            ('higgs_v3_voice_text', 'higgs_v3_voice_transcript'),
            ('chatterbox_ml_repetition_penalty', 'chatterbox_ml_v2_repetition_penalty'),
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
            dialog_segmentation=bool(project_settings.get('applied_dialog_segmentation', False)),
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
