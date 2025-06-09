import gc
import json
import logging
import os
import glob
import tempfile
from typing import Any
import glob
import torch

from tts_audiobook_tool.l import L

from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class AppUtil:

    _is_logging_initialized = False

    @staticmethod
    def init_logging() -> None:

        if AppUtil._is_logging_initialized:
            return
        AppUtil._is_logging_initialized = True

        L.init(APP_NAME)
        L.i("START " + "-" * 60)
        # Squelch various 3p lib output
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        # Used by oute lib
        from loguru import logger
        logger.remove()
        import warnings
        warnings.filterwarnings("ignore", module="pyloud")

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
    def print_text_segment_text(raw_texts: list[str]) -> None:
        print_heading(f"Text segments ({COL_DEFAULT}{len(raw_texts)}{COL_ACCENT}):")
        for i, raw_text in enumerate(raw_texts):
            printt(f"{make_hotkey_string(str(i+1))} {raw_text.strip()}")
        printt()

    @staticmethod
    def gc_ram_vram() -> None:
        # Force-trigger Python garbage collector
        gc.collect()
        # "Garbage collect" VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_app_temp_dir() -> str:
        app_temp_dir = os.path.join(tempfile.gettempdir(), APP_TEMP_SUBDIR)
        Path(app_temp_dir).mkdir(exist_ok=True) # not catching error here
        return app_temp_dir

    @staticmethod
    def get_temp_file_path_by_hash(hash: str) -> str:
        """
        Returns file path of item in the app temp directory or empty string
        """
        dir = AppUtil.get_app_temp_dir()
        if not dir:
            return ""
        items = os.listdir(dir)
        for item in items:
            if hash in item:
                return os.path.join(dir, item)
        return ""

    @staticmethod
    def show_player_reminder(prefs: Prefs) -> None:

        printt(f"🔔 {COL_ACCENT}Reminder:")
        printt("You can open audio files with the interactive player/reader here:")
        package_dir = get_package_dir()
        if package_dir:
            browser_path = str( Path(package_dir).parent / "browser_player" / "index.html" )
        else:
            browser_path = "browser_player" + os.path.sep + "index.html"
        printt(browser_path)
        printt(f"or on the web here: {PLAYER_URL}")
        printt()

        prefs.has_shown_player_reminder = True
