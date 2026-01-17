import os
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *

class ProjectUtil:
    """
    Helper functions for `Project`
    """
    
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
    def get_latest_concat_files(project: Project, limit=10) -> list[str]:
        """
        Finds the latest concatenated audio files within the project's concat directory.
        """
        concat_dir = project.concat_path
        if not concat_dir or not os.path.exists(concat_dir):
            return []
        
        # Get subdirectories, revchron sorted
        subdirs = []
        for item in os.listdir(concat_dir):
            item_path = os.path.join(concat_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item_path)        
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
        # Iterate through subdirectories
        DEBUG_TAGS = ["[concat]", "[norm]", "[chaptermeta]"] 
        paths = []
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
                    paths.append(path)
                    break
            if len(paths) > limit:
                break
        
        return paths
