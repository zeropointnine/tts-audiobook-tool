import logging
import os
import glob
import tempfile
import xxhash

from tts_audiobook_tool.app_types import ModelWarmUpResult, SegmentationStrategy, SttVariant
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.hint_util import HintUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
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

        # Squelch various noisey 3p lib output
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
        
        # TTS libs (target only noisey parts)
        logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)        
        import warnings
        warnings.filterwarnings("ignore", module="pyloud") # used by oute

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
        MenuUtil.print_heading(None, s, non_menu=True)
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

        heading = "Text segments:" if extant_indices else "Text segments preview:"
        MenuUtil.print_heading(None, heading, dont_clear=True)

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

        printt(f"{COL_DIM}The text was segmented using the following settings:")
        printt(f"- Text segmenter language code: {COL_ACCENT}{language_code_used or 'none'}")
        if max_words_used:
            printt(f"- Text segmenter max_words_per_segment: {COL_ACCENT}{max_words_used}")
        printt(f"- Text segmenter strategy: {COL_ACCENT}{strategy_used.label}")
        if extant_indices is not None:
            printt()
            printt(f"Num audio segments generated: {COL_ACCENT}{len(extant_indices)} {COL_DIM}/ {COL_ACCENT}{len(phrase_groups)}")
        printt()                

    @staticmethod
    def print_regen_lines(project: Project, indices: set[int]) -> None:
        from tts_audiobook_tool.segment_stt_info_util import SegmentSttInfoUtil
        
        MenuUtil.print_heading(None, "Lines to be regenerated:", non_menu=True, dont_clear=True)

        if not indices:
            printt("None")
            printt()
            return

        for index in sorted(indices):
            SegmentSttInfoUtil.print_info(index, project)

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
    def set_seed(seed: int):
        """ 
        Sets random seed across torch, numpy, and random. 
        
        This is oftentimes sufficient to make the inference of torch-based TTS models idempotent.
        (Must be tested carefully to verify, ofc)
        """
        import torch
        import numpy as np
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

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
        """ Shows one-time hints or warnings related to doing inference """
        
        from tts_audiobook_tool.tts import Tts
        from tts_audiobook_tool.project import Project
        project: Project = p_project

        if Tts.get_type() == TtsModelInfos.FISH_S1 and project.fish_s1_compile_enabled:
            HintUtil.show_hint_if_necessary(prefs, HINT_FISH_S1_FIRST_COMPILE)
        elif Tts.get_type() == TtsModelInfos.FISH_S2 and project.fish_s2_compile_enabled:
            HintUtil.show_hint_if_necessary(prefs, HINT_FISH_S2_FIRST_COMPILE)

        import torch
        if platform.system() == "Linux" and torch.cuda.is_available():
                if prefs.stt_variant != SttVariant.DISABLED and prefs.stt_config.device == "cuda":
                    version = torch.backends.cudnn.version()
                    if version and version > CTRANSLATE_REQUIRED_CUDNN_VERSION:
                        HintUtil.show_hint(HINT_LINUX_CUDNN_VERSION, and_prompt=True)

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
    def play_done_sound() -> None:
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        done_wav_path = make_assets_file_path("done.wav")
        SoundFileUtil.play_sound_file_async(done_wav_path)

    @staticmethod
    def make_memory_string(base_color=COL_DIM) -> str:
        
        from tts_audiobook_tool.memory_util import MemoryUtil
        
        result = MemoryUtil.get_nv_vram()
        if result is None:
            vram_string = ""
        else:
            used, total = result
            vram_string = f"{base_color}VRAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

        shared_result = MemoryUtil.get_shared_gpu_memory()
        if shared_result is None:
            shared_string = ""
        else:
            _, shared = shared_result
            shared_string = f"{base_color}shared VRAM: {COL_ACCENT}{make_gb_string(shared)}"

        result = MemoryUtil.get_system_ram()
        if result is None:
            ram_string = ""
        else:
            used, total = result
            ram_string = f"{base_color}RAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

        parts = [s for s in [vram_string, shared_string, ram_string] if s]
        if not parts:
            return ""
        return f"{base_color}, ".join(parts)

    @staticmethod
    def print_warm_up_result_stop(result: ModelWarmUpResult) -> None:
        """ Prints user-facing feedback for a failed warm-up result. """
        if result.did_interrupt:
            print_feedback("\nCancelled")
        elif result.error:
            print_feedback(f"\n{COL_ERROR}Model initialization failed:\n\n{result.error}")

    @staticmethod
    def log_unload_memory_snapshot(label: str) -> None:
        parts: list[str] = []

        import torch

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_allocated = torch.cuda.max_memory_allocated()
                max_reserved = torch.cuda.max_memory_reserved()
                parts.extend([
                    f"torch_allocated={make_gb_string(allocated)}",
                    f"torch_reserved={make_gb_string(reserved)}",
                    f"torch_max_allocated={make_gb_string(max_allocated)}",
                    f"torch_max_reserved={make_gb_string(max_reserved)}",
                ])

                stats = torch.cuda.memory_stats()

                def add_stat_bytes(label_text: str, stat_key: str) -> None:
                    value = stats.get(stat_key)
                    if value is not None:
                        parts.append(f"{label_text}={make_gb_string(int(value))}")

                def add_stat_int(label_text: str, stat_key: str) -> None:
                    value = stats.get(stat_key)
                    if value is not None:
                        parts.append(f"{label_text}={int(value)}")

                add_stat_bytes("active_bytes", "active_bytes.all.current")
                add_stat_bytes("inactive_split_bytes", "inactive_split_bytes.all.current")
                add_stat_bytes("requested_bytes", "requested_bytes.all.current")
                add_stat_bytes("allocated_bytes", "allocated_bytes.all.current")
                add_stat_bytes("reserved_bytes", "reserved_bytes.all.current")
                add_stat_int("active_blocks", "active.all.current")
                add_stat_int("segment_count", "segment.all.current")
            except Exception as e:
                parts.append(f"torch_cuda_error={e}")
        else:
            parts.append("torch_cuda_unavailable")

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pid = os.getpid()
            process_bytes = None

            try:
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc.pid == pid:
                        process_bytes = int(proc.usedGpuMemory)
                        break
            except Exception as e:
                parts.append(f"nvml_compute_proc_error={e}")

            try:
                for proc in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                    if proc.pid == pid:
                        used = int(proc.usedGpuMemory)
                        process_bytes = used if process_bytes is None else process_bytes + used
                        break
            except Exception as e:
                parts.append(f"nvml_graphics_proc_error={e}")

            if process_bytes is None:
                parts.append(f"pid_gpu_memory=none pid={pid}")
            else:
                parts.append(f"pid_gpu_memory={make_gb_string(process_bytes)} pid={pid}")

            pynvml.nvmlShutdown()
        except Exception as e:
            parts.append(f"nvml_error={e}")

        L.i(f"Unload memory snapshot [{label}] | " + " | ".join(parts))
