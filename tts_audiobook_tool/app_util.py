import logging
import os
import glob
import tempfile
import glob
import xxhash

from tts_audiobook_tool.app_types import SegmentationStrategy, SttVariant
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.validate_util import ValidateUtil

class AppUtil:
    """
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
    def print_text_groups(groups: list[PhraseGroup]) -> None:
        s = f"Text segments ({COL_DIM}{len(groups)}{COL_DEFAULT}):"
        print_heading(s, non_menu=True)
        printt()

        for i, group in enumerate(groups):
            printt(f"{make_hotkey_string(str(i+1))} {group.presentable_text}")
        printt()

    @staticmethod
    def print_project_text(
            phrase_groups: list[PhraseGroup], 
            extant_indices: set[int] | None,
            language_code_used: str,
            max_words_used: int | None,
            strategy_used: SegmentationStrategy
    ) -> None:
        """
        Prints the list of text segments of a Project.

        extant_indices:
            When exists, prints if sound gen exists for text segment, and prints num-generated info
        """        

        heading = "Text segments" if extant_indices else "Text segments preview"
        print_heading(heading, non_menu=True)

        if len(phrase_groups) > 0:
            index_width = len(str(len(phrase_groups)))
            
            for i, phrase_group in enumerate(phrase_groups):
                index_string = "[" + str(i+1).rjust(index_width) + "]"
                if extant_indices is not None:
                    exists_string = "[" + ("generated" if i in extant_indices else " missing ") + "] "
                else:
                    exists_string = ""
                if DEV:
                    reason_string = COL_DIM + " [" + phrase_group.as_flattened_phrase().reason.json_value + "]"
                else:
                    reason_string = ""
                printt(f"{COL_ACCENT}{index_string} {COL_DIM}{exists_string}{COL_DEFAULT}{phrase_group.presentable_text}{reason_string}")        
        else:
            printt("None")    
        printt()

        if extant_indices is not None:
            printt(f"- Num audio segments generated: {COL_ACCENT}{len(extant_indices)} {COL_DIM}/ {COL_ACCENT}{len(phrase_groups)}")
        printt(f"- Text segmenter language code: {COL_ACCENT}{language_code_used or 'none'}")
        if max_words_used:
            printt(f"- Text segmenter max_words_per_segment: {COL_ACCENT}{max_words_used}")
        printt(f"- Text segmenter strategy: {COL_ACCENT}{strategy_used.label}")
        printt()                

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
    def calc_hash_string(string: str) -> str:
        return xxhash.xxh3_64(string).hexdigest()

    @staticmethod
    def calc_hash_file(path: str, print_progress_text: str="") -> tuple[str, str]:
        """ Returns hash and error string, mutually exclusive"""

        if not os.path.exists(path):
            return "", f"File not found: {path}"
        if os.path.isdir(path):
            return "", f"Is not a file: {path}"
        if not os.access(path, os.R_OK):
            return "",  f"No read permission for file: {path}"

        if print_progress_text:
            print_progress_text = print_progress_text.strip()

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
                    if print_progress_text:
                        print(f"\r{print_progress_text} {processed/file_size:.1%} ", end='')

        except Exception as e:
            return "", f"Error while hashing file: {e}"

        if print_progress_text:
            print("\r", end = "")  # Clear progress text

        return hasher.hexdigest(), ""

    @staticmethod
    def is_app_hash(hash: str) -> bool:
        """ The app uses 16-character hex string for hash values """
        return len(hash) == 16 and all(c in '0123456789abcdefABCDEF' for c in hash)

    @staticmethod
    def path_insert_tag(file_path: str, tag: str) -> str:
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
    def show_pre_inference_hints(prefs: Prefs, p_project) -> None:
        """ Shows one-time hints related to doing inference """

        from tts_audiobook_tool.project import Project
        project: Project = p_project

        if Tts.get_type() == TtsModelInfos.FISH:
            Hint.show_hint_if_necessary(prefs, HINT_FISH_FIRST)

        if project.can_voice and project.get_voice_label() == "none":
            Hint.show_hint_if_necessary(prefs, HINT_NO_VOICE)

        import torch
        if platform.system() == "Linux" and torch.cuda.is_available():
                if prefs.stt_variant != SttVariant.DISABLED and prefs.stt_config.device == "cuda":
                    version = torch.backends.cudnn.version()
                    if version and version > CTRANSLATE_REQUIRED_CUDNN_VERSION:
                        Hint.show_hint(HINT_LINUX_CUDNN_VERSION, and_prompt=True)

    @staticmethod
    def should_stt(state: State) -> bool:
        """
        Should the app be doing speech-to-text transcription of the generated audio?
        """
        if ValidateUtil.is_unsupported_language_code(state.project.language_code):
            return False
        if state.prefs.stt_variant == SttVariant.DISABLED:
            return False
        return True

    @staticmethod
    def get_chrome_path() -> str:
        """ Tries to find Chrome or Chromium path on local machine """
        
        import platform
        
        system = platform.system()
        
        if system == "Windows":
            # Common Chrome paths on Windows
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
            ]
        elif system == "Darwin":  # macOS
            # Common Chrome/Chromium paths on macOS
            chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
            ]
        elif system == "Linux":
            # Common Chrome/Chromium paths on Linux
            chrome_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/opt/google/chrome/google-chrome",
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium-browser-stable",
                "/snap/bin/chromium",
            ]
        else:
            # Unsupported OS
            return ""
        
        # Check each path and return the first one that exists
        for path in chrome_paths:
            if os.path.exists(path):
                return path
        
        return ""
