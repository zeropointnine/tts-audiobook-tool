import json
import os
import glob
from typing import Any

from tts_audiobook_tool.l import L

from .util import *

class AppUtil:

    @staticmethod
    def sanitize_for_filename(filename) -> str:
        """
        Replaces all non-alpha-numeric characters with underscores,
        replaces consecutive underscores with a single underscore,
        and removes leading/trailing underscores.
        """
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', filename)    
        collapsed = re.sub(r'_+', '_', sanitized)
        stripped = collapsed.strip('_')    
        return stripped


    @staticmethod
    def delete_project_audio_files(dir: str) -> str:
        """
        Deletes all FLAC files in the specified directory.
        Returns user-facing error message string on fail else empty string on success

        not using for now
        """
        num_fail = 0
        flac_files = glob.glob(os.path.join(dir, '*.flac'))

        for file_path in flac_files:
            try:
                os.remove(file_path)
            except OSError:
                num_fail += 1

        if num_fail > 0:
            return f"Warning: Couldn't delete {num_fail} now-outdated audio segment(s)."
        else:
            return ""

    @staticmethod
    def save_json(json_object: Any, path: str) -> str:
        """
        Returns error message on fail, else empty string
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)
                return ""
        except Exception as e:
            return f"Error saving json: {e}"            
