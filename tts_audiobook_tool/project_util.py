import os
from datetime import datetime
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *

class ProjectUtil:
    """
    Helper functions for `Project`
    """
    
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

