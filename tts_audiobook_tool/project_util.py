from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_models.oute_util import OuteUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project

class ProjectUtil:
    """
    Helper functions for `Project`
    """
    
    @staticmethod
    def load_using_dir_path(dir_path: str) -> Project | str:
        """
        Loads project json from directory path and returns parsed project instance.
        Returns error string if json is unviable.
        Else, on some parse errors, falls back to defaults and prints info along the way.
        """
        from tts_audiobook_tool.ask_util import AskUtil
        from tts_audiobook_tool.l import L
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

        d['dir_path'] = dir_path  # inject; not stored in JSON

        inline_text_source = ""
        if "text" in d:
            inline_text_source = "text"
        elif "text_segments" in d:
            inline_text_source = "text_segments"
        elif 'phrase_groups' not in d:
            result = ProjectUtil.load_phrase_groups_payload(dir_path)
            if isinstance(result, str):
                return result
            if result is not None:
                d['phrase_groups'] = result

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

        if Tts.get_type() == TtsModelInfos.OUTE:
            ProjectUtil.load_oute_voice_json(project)

        did_clear_invalid_voice_files = project.verify_voice_files_exist()

        pending = getattr(thread_local, 'warnings', [])
        thread_local.warnings = []
        if pending or did_clear_invalid_voice_files:
            project.save()
            for warning in pending:
                printt(warning)
            AskUtil.ask_enter_to_continue()

        project._autosave = True
        return project

    @staticmethod
    def is_valid_project_dir(project_dir: str) -> str:
        """ Returns error feedback text or empty string if is-valid """

        if not os.path.exists(project_dir):
            return f"Doesn't exist: {project_dir}"

        items = os.listdir(project_dir)

        # Empty directory is considered valid
        if not items:
            return ""

        # Directory with a voice and/or text json file considered valid
        if PROJECT_JSON_FILE_NAME in items:
            return ""

        return f"{project_dir} does not appear to be a project directory"

    @staticmethod
    def parse_word_substitutions_json_string(inp: str) -> dict[str, str] | str:
        """ Returns list of two-string lists or user-facing error message """
        try:
            import ast
            dic = ast.literal_eval(inp) # less strict than json decode, which requires double-quotes
        except Exception as e:
            return make_error_string(e)
        if not isinstance(dic, dict):
            return "Bad type"
        if len(dic) == 0:
            return "No items entered"
        for key, value in dic.items():
            if not key:
                return f"Found empty string"
            if not value:
                return f"Found empty string for {key}"
            if not isinstance(key, str):
                return f"Not a string: {key}"
            if not isinstance(value, str):
                return f"Not a string: {value}"
            # fyi, not checking for extra-long strings, punctuation, etc

        # Massage
        result = {}
        for key, value in dic.items():
            key = key.lower().strip() # search-word is treated as case-insensitive
            value = value.strip() # replacement-word is not
            result[key] = value
        return result

    @staticmethod
    def remap_legacy_keys(d: dict) -> None:
        """Mutates d to normalize old JSON key names to current names."""
        for old, new in [
            ('fish_voice_file_name', 'fish_s1_voice_file_name'),
            ('fish_voice_text', 'fish_s1_voice_text'),
            ('fish_temperature', 'fish_s1_temperature'),
            ('fish_seed', 'fish_s1_seed'),
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

    @staticmethod
    def load_oute_voice_json(project: Project) -> None:
        """Load Oute voice JSON into project.oute_voice_json (only when Oute is active)."""
        voice_path = os.path.join(project.dir_path, project.oute_voice_file_name)
        if not project.oute_voice_file_name or not os.path.exists(voice_path):
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                from tts_audiobook_tool.ask_util import AskUtil
                AskUtil.ask_error(result)
            else:
                project.set_oute_voice_and_save(result, "default")
        else:
            result = OuteUtil.load_oute_voice_json(voice_path)
            if isinstance(result, str):
                printt(f"Problem loading Oute voice json file {project.oute_voice_file_name}: {result}")
                printt()
            else:
                project.oute_voice_json = result

    @staticmethod
    def get_indices_to_generate(project: Project) -> set[int]:
        """
        Returns the set of indices to be generated,
        derived from the (human readable) "generate_range_string"
        """
        range_string = project.generate_range_string
        is_all = not range_string or range_string == "all" or range_string == "a"
        if is_all:
            result = set(range(len(project.phrase_groups)))
        else:
            result, _ = ParseUtil.parse_ranges_string(range_string, len(project.phrase_groups))
        return result

    @staticmethod
    def get_selected_indices_not_generated(project: Project) -> set[int]:
        """
        From the currently selected range of indices,
        returns the indicies for which no sound segment exists.
        """
        selected_indices_all = ProjectUtil.get_indices_to_generate(project)
        selected_indices_generated = set(project.sound_segments.sound_segments_map.keys())
        selected_indices_not_generated = selected_indices_all - selected_indices_generated
        return selected_indices_not_generated
    
    @staticmethod
    def get_latest_concat_files(project: Project, limit=10) -> list[tuple[str, str]]:
        """
        Finds latest concatenated audio files within the project's concat directory,
        returning tuples of (path, modified_date) where modified_date is
        YYYY-MM-DD HH:MM.
        """
        concat_dir = project.concat_path
        if not concat_dir or not os.path.exists(concat_dir):
            return []
        
        # Get subdirectories, revchron-sorted
        subdirs = []
        for item in os.listdir(concat_dir):
            item_path = os.path.join(concat_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item_path)        
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
        # Iterate through subdirectories
        DEBUG_TAGS = ["[concat]", "[norm]", "[chaptermeta]"] 
        results = []
        for subdir in subdirs:
            try:
                file_names = sorted(os.listdir(subdir)) # alpha-sorted
            except (OSError, PermissionError):
                continue
            for file_name in file_names:
                hit = file_name.endswith((".abr.m4b", ".abr.flac")) 
                hit = hit and not any(sub in file_name for sub in DEBUG_TAGS)
                if hit:
                    path = os.path.join(subdir, file_name)
                    try:
                        modified_date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
                    except (OSError, PermissionError, ValueError):
                        modified_date = "unknown"
                    results.append((path, modified_date))
                    break
            if len(results) >= limit:
                break
        
        return results

    @staticmethod
    def parse_emo_vector_string(string: str) -> list[float] | str:
        """
        Returns error string on parse fail
        Returns empty list to represent list of all zeroes
        """

        string = string.strip()
        if not string:
            return []
        if string.lower() == "none":
            return []

        strings = string.split(",")
        if len(strings) != 8:
            return "Requires 8 comma-delimited numbers between 0-1"

        floats = []
        for string in strings:
            try:
                flt = float(string)
            except:
                return f"Bad value: {string} - must be a number between 0-1"
            if not (0 <= flt <= 1):
                return f"Out of range: {flt} - must be between 0-1"
            floats.append(flt)
        return floats

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
        skip = {'dir_path', 'sound_segments', 'oute_voice_json'}
        with dest_project.batch():
            for field_name in source_project.model_fields:
                if field_name not in skip:
                    setattr(dest_project, field_name, getattr(source_project, field_name))

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
            
            src_path = ProjectUtil.find_supporting_project_file_source_path(source_dir, file_name)
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
        
        candidate_names = [file_name]

        if file_name == PROJECT_TEXT_RAW_FILE_NAME:
            candidate_names.append("text_raw.txt")

        for candidate_name in candidate_names:
            candidate_path = os.path.join(source_dir, candidate_name) if source_dir else candidate_name
            if os.path.exists(candidate_path):
                return candidate_path

        return ""

