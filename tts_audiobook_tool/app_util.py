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
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class AppUtil:
    """
    """

    _is_logging_initialized = False

    @staticmethod
    def init_logging() -> None:

        if AppUtil._is_logging_initialized:
            return
        AppUtil._is_logging_initialized = True

        # Init app logger
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
    def get_chromium_user_data_dir() -> str:
        dir = AppUtil.get_app_user_dir()
        dir = os.path.join(dir, CHROME_USER_DATA_DIR_NAME)
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
    def local_path_for_display(path: str) -> str:
        """ App style for displaying local path in menu/UI """
        LEN = 60
        if len(path) <= LEN:
            return path
        p = Path(path)
        # Find the separator before p.name and construct truncated path
        parent_str = str(p.parent)
        sep_pos = parent_str.rfind(os.sep)
        if sep_pos >= 0:
            # Include the separator and everything after it
            suffix = parent_str[sep_pos:] + os.sep + p.name
        else:
            suffix = p.name
        # Calculate how many chars we can take from suffix
        prefix_len = 10  # first 10 chars
        ellipsis_len = 3  # "..."
        max_suffix_len = LEN - prefix_len - ellipsis_len
        if len(suffix) > max_suffix_len:
            suffix = suffix[-max_suffix_len:]
        return path[:prefix_len] + "..." + suffix
        
    @staticmethod
    def show_pre_inference_hints(prefs: Prefs, p_project) -> None:
        """ Shows one-time hints related to doing inference """
        
        from tts_audiobook_tool.tts import Tts
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
    def get_chromium_info() -> tuple[str, str] | None:
        """ 
        Looks for Chromium-based browser on local machine, 
        and returns app name and path.
        """
        import platform
        system = platform.system()
        
        if system == "Windows":
            names_and_paths = [
                ("Chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                ("Chrome", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
                ("Chrome", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe")),
                ("Chromium", r"C:\Program Files\Chromium\Application\chrome.exe"),
                ("Chromium", r"C:\Program Files (x86)\Chromium\Application\chrome.exe"),
                ("Chromium", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Chromium", "Application", "chrome.exe")),
                ("Edge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            ]
        elif system == "Darwin": # macOS
            names_and_paths = [
                ("Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                ("Chromium", "/Applications/Chromium.app/Contents/MacOS/Chromium"),
            ]
        elif system == "Linux":
            names_and_paths = [
                ("Chrome", "/usr/bin/google-chrome"),
                ("Chrome", "/usr/bin/google-chrome-stable"),
                ("Chrome", "/opt/google/chrome/google-chrome"),
                ("Chromium", "/usr/bin/chromium"),
                ("Chromium", "/usr/bin/chromium-browser"),
                ("Chromium", "/usr/bin/chromium-browser-stable"),
                ("Chromium", "/snap/bin/chromium"),
            ]
        else:
            return None
        
        # Check each path and return the first one that exists
        for item in names_and_paths:
            if os.path.exists(item[1]):
                return item
        
        return None

    @staticmethod
    def does_hf_model_exist(
            path_or_id: str,
            local_file_markers = ["config.json", "model.safetensors", "adapter_config.json"]
    ) -> tuple[str, str]:
        """
        `path_or_id` can be either a local directory or a huggingface repo id

        Returns 
            Tuple with source type (local_path | hf_cache | hf_remote), UI error string 
            (mutually exclusive)
        """

        from huggingface_hub import model_info, try_to_load_from_cache
        from huggingface_hub.errors import RepositoryNotFoundError, HFValidationError
        from huggingface_hub.utils._validators import validate_repo_id

        # If value is an existing local directory, check for existence a 'local file marker'        
        if os.path.exists(path_or_id):
            if not os.path.isdir(path_or_id):
                return "", "Path is not a directory"
            is_hf_dir = any(os.path.isfile(os.path.join(path_or_id, marker)) for marker in local_file_markers)
            if is_hf_dir:
                return "local_path", ""
            else:
                return "", "Directory is missing typical hf file markers"
            
        error = "No such local directory;\n"

        # Test for correct hf repo id string format
        try:
            validate_repo_id(path_or_id)
        except HFValidationError:
            error += "Invalid string format for hf repo id"
            return "", error

        # Check local hf cache
        for file_marker in local_file_markers:
            cached_path = try_to_load_from_cache(path_or_id, file_marker)
            if isinstance(cached_path, str):
                return "hf_cache", ""
        
        # Test for hf repo remote reachability
        try:
            model_info(path_or_id)
            return "hf_remote", ""
        except RepositoryNotFoundError:
            error += "Repo not found or gated"
            return "", error
        except Exception as e:
            error += f"Network or other error: {e}"
            return "", error

    @staticmethod
    def make_memory_string(base_color=COL_DIM) -> str:
        
        from tts_audiobook_tool.memory_util import MemoryUtil
        
        result = MemoryUtil.get_nv_vram()
        if result is None:
            vram_string = ""
        else:
            used, total = result
            vram_string = f"{base_color}VRAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

        result = MemoryUtil.get_system_ram()
        if result is None:
            ram_string = ""
        else:
            used, total = result
            ram_string = f"{base_color}RAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

        if not vram_string and not ram_string:
            return ""
        elif not vram_string:
            return ram_string
        elif not ram_string:
            return vram_string
        else:
            return f"{vram_string}{base_color}, {ram_string}"
