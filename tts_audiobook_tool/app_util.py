import json
import logging
import os
import glob
from typing import Any
import glob
from datetime import datetime
from mutagen.flac import FLAC

from tts_audiobook_tool.l import L

from tts_audiobook_tool.util import *

class AppUtil:

    @staticmethod
    def init_app_logging() -> None:
        L.init(APP_NAME)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)

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
    def make_timestamp_string() -> str:
        current_time = datetime.now()
        return current_time.strftime("%y%m%d_%H%M%S")

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

    @staticmethod
    def get_flac_file_duration(path: str) -> float | None:
        """
        Returns the duration in seconds of a FLAC file, or None
        """
        try:
            audio = FLAC(path)
            if audio.info:
                return audio.info.length
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def time_string(seconds: float) -> str:
        """ 5h0m0s """
        seconds = round(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        seconds = seconds % 60
        if minutes < 60:
            return f"{minutes}m{seconds}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h{minutes}m{seconds}s"

    @staticmethod
    def print_text_segments(texts: list[str]) -> None:
        printt(f"{COL_ACCENT}Text segments ({COL_DEFAULT}{len(texts)}{COL_ACCENT}):\n")
        for i, segment in enumerate(texts):
            printt(f"{make_hotkey_string(str(i))} {segment}")
        printt()

