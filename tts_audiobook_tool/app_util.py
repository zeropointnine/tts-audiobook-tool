import gc
import logging
import os
import glob
import sys
import tempfile
import glob
import time
import torch
import xxhash

from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.text_segmenter import TextSegmenter
from tts_audiobook_tool.util import *

class AppUtil:
    """
    Need separation btw 'low level' and 'high level' functions
    """

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
        import warnings
        warnings.filterwarnings("ignore", module="pyloud")

    @staticmethod
    def delete_project_sound_files(dir: str) -> str:
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
    def print_text_segment_text(raw_texts: list[str]) -> None:

        printt(f"{COL_ACCENT}Text segments ({COL_DEFAULT}{len(raw_texts)}{COL_ACCENT}):")
        printt()

        for i, raw_text in enumerate(raw_texts):
            printt(f"{make_hotkey_string(str(i+1))} {raw_text.strip()}")

        printt()

    @staticmethod
    def gc_ram_vram() -> None:
        """ Trigger Python garbage collector, plus torch"""
        gc.collect()
        # "Garbage collect" VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_app_temp_dir() -> str:
        # eg, 'C:\Users\me\AppData\Local\Temp\tts_audiobook_tool'
        dir = os.path.join(tempfile.gettempdir(), APP_TEMP_SUBDIR)
        Path(dir).mkdir(exist_ok=True) # not catching error here
        return dir

    @staticmethod
    def get_app_user_dir() -> str:
        # eg, 'C:\Users\me\tts_audiobook_tool'
        dir = os.path.join(Path.home(), APP_USER_SUBDIR)
        Path(dir).mkdir(exist_ok=True) # not catching error here
        return dir

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
    def get_text_segments_from_ask_text_file() -> tuple[ list[TextSegment], str ]:
        """
        Asks user for path to text file and returns list of TextSegments and raw text.
        Shows feedback except when text segments are returned
        """
        path = AskUtil.ask_file_path("Enter text file path: ", "Select text file")
        if not path:
            return [], ""
        if not os.path.exists(path):
            AskUtil.ask_error("No such file")
            return [], ""

        try:
            with open(path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
        except Exception as e:
            AskUtil.ask_error(f"Error: {e}")
            return [], ""

        print("Segmenting text... ", end="", flush=True)
        text_segments = TextSegmenter.segment_text(raw_text, max_words=MAX_WORDS_PER_SEGMENT)
        print(f"\r{Ansi.ERASE_REST_OF_LINE}", end="", flush=True)

        if not text_segments:
            AskUtil.ask_enter_to_continue("No text segments.")
            return [], raw_text

        return text_segments, raw_text

    @staticmethod
    def get_text_segments_from_ask_std_in() -> tuple[ list[TextSegment], str ]:
        """
        Asks user to input text using stdin.read() and returns list of TextSegments and raw text
        """
        printt("Enter/paste text of any length.")
        if platform.system() == "Windows":
            s = f"Finish with {COL_ACCENT}[CTRL-Z + ENTER] {COL_DEFAULT}on its own line"
        else:
            s = f"Finish with [ENTER + CTRL-D]"
        printt(s)
        printt()
        raw_text = sys.stdin.read().strip()
        printt()
        if not raw_text:
            return [], ""

        text_segments = TextSegmenter.segment_text(raw_text, max_words=MAX_WORDS_PER_SEGMENT)

        if not text_segments:
            AskUtil.ask_enter_to_continue("No text segments.")
            return [], raw_text

        return text_segments, raw_text

    @staticmethod
    def show_hint_if_necessary(prefs: Prefs, hint: Hint, and_confirm: bool=False, and_prompt: bool=False) -> bool:
        """
        If hint has already been shown does nothing.
        Else, shows hint.
        Then either asks for confirmation, prompts to press enter, or or shows a 3-second 'animation'
        Returns True if "should continue"
        """
        if prefs.get_hint(hint.key):
            return True
        prefs.set_hint_true(hint.key)

        AppUtil.print_hint(hint)

        if and_confirm:
            return AskUtil.ask_confirm()
        elif and_prompt:
            AskUtil.ask_enter_to_continue()
            return True
        else:
            # Anim
            lines = ["[   ]", "[.  ]", "[.. ]", "[...]"]
            for i, line in enumerate(lines):
                print(f"{COL_DIM}{line}{Ansi.RESET}", end="\r", flush=True)
                time.sleep(0.66)
            print(f"{Ansi.ERASE_REST_OF_LINE}", end="", flush=True)
            return True

    @staticmethod
    def print_hint(hint: Hint) -> None:
        printt(f"🔔 {COL_ACCENT}{hint.heading}")
        printt(hint.text)
        printt()

    @staticmethod
    def show_player_hint_if_necessary(prefs: Prefs) -> None:
        s = "You can open audio files with the interactive player/reader here:\n"
        package_dir = get_package_dir()
        if package_dir:
            browser_path = str( Path(package_dir).parent / "browser_player" / "index.html" )
        else:
            browser_path = "browser_player" + os.path.sep + "index.html"
        s += browser_path + "\n"
        s += "or on the web here:" + "\n"
        s += PLAYER_URL

        hint = Hint(key="player", heading="Reminder", text = s)
        AppUtil.show_hint_if_necessary(prefs, hint)

    @staticmethod
    def calc_hash_string(string: str) -> str:
        return xxhash.xxh3_64(string).hexdigest()

    @staticmethod
    def calc_hash_file(path: str, with_progress: bool=False) -> tuple[str, str]:
        """ Returns hash and error string, mutually exclusive"""

        if not os.path.exists(path):
            return "", f"File not found: {path}"
        if os.path.isdir(path):
            return "", f"Is not a file: {path}"
        if not os.access(path, os.R_OK):
            return "",  f"No read permission for file: {path}"

        hasher = xxhash.xxh64()
        file_size = os.path.getsize(path)
        processed = 0

        try:
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    processed += len(chunk)
                    print(f"\rHashing file: {processed/file_size:.1%}", end='')

        except Exception as e:
            return "", f"Error while hashing file: {e}"

        print("\r", end = "")  # Clear progress text

        return hasher.hexdigest(), ""

    @staticmethod
    def is_app_hash(hash: str) -> bool:
        """ The app uses 16-character hex string for hash values """
        return len(hash) == 16 and all(c in '0123456789abcdefABCDEF' for c in hash)

    @staticmethod
    def insert_bracket_tag_file_path(file_path: str, tag: str) -> str:
        """
        Eg, "[one] [two] hello.flac" -> "[one] [two] [newtag] hello.flac"
        """
        path = Path(file_path)
        stem = path.stem
        i = stem.rfind("]") + 1
        substring = f"[{tag}]"
        if i > 0:
            substring = " " + substring
        else:
            substring = substring + " "
        new_stem = stem[:i] + substring + stem[i:]
        new_path = path.with_stem(new_stem)
        return str(new_path)

    @staticmethod
    def get_nv_vram() -> tuple[int, int] | None:
        """
        Returns VRAM bytes used, bytes total from NVIDIA GPU (device 0), or None if no nvidia gpu
        Requires nvidia-ml-py (pip install nvidia-ml-py)
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return int(info.used), int(info.total)
        except Exception as e:
            return None
