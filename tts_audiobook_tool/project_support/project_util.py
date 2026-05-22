from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project

__all__ = ["ProjectUtil", "Tts"]


class ProjectUtil:
    """
    Small grab-bag of project-related helpers.
    """

    @staticmethod
    def load_using_dir_path(dir_path: str):
        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
        return ProjectLoadUtil.load_using_dir_path(dir_path)

    @staticmethod
    def is_valid_project_dir(project_dir: str) -> str:
        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
        return ProjectLoadUtil.is_valid_project_dir(project_dir)

    @staticmethod
    def remap_legacy_keys(d: dict) -> None:
        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
        ProjectLoadUtil.remap_legacy_keys(d)

    @staticmethod
    def load_book_payload(project_dir: str, project_settings: dict):
        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
        return ProjectLoadUtil.load_book_payload(project_dir, project_settings)

    @staticmethod
    def load_phrase_groups_payload(dir_path: str):
        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
        return ProjectLoadUtil.load_phrase_groups_payload(dir_path)

    @staticmethod
    def load_raw_abr_metadata_string(abr_path: str) -> str:
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        return ProjectTransferUtil.load_raw_abr_metadata_string(abr_path)

    @staticmethod
    def make_project_from_snapshot(project_dir: str, project_snapshot: dict):
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        return ProjectTransferUtil.make_project_from_snapshot(project_dir, project_snapshot)

    @staticmethod
    def apply_project_settings(dest_project: Project, source_project: Project) -> None:
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        ProjectTransferUtil.apply_project_settings(dest_project, source_project)

    @staticmethod
    def get_snapshot_source_dir(project_snapshot: dict) -> str:
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        return ProjectTransferUtil.get_snapshot_source_dir(project_snapshot)

    @staticmethod
    def make_supporting_project_file_names(project: Project) -> list[str]:
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        return ProjectTransferUtil.make_supporting_project_file_names(project)

    @staticmethod
    def copy_supporting_project_files(project: Project, source_dir: str, file_names: list[str]) -> list[str]:
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        return ProjectTransferUtil.copy_supporting_project_files(project, source_dir, file_names)

    @staticmethod
    def find_supporting_project_file_source_path(source_dir: str, file_name: str) -> str:
        from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
        return ProjectTransferUtil.find_supporting_project_file_source_path(source_dir, file_name)

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
            result, _ = RangeStringUtil.parse_ranges_string(range_string, len(project.phrase_groups))
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
        ONE_FILE_PER_SUBDIR = True

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
                    if ONE_FILE_PER_SUBDIR:
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


