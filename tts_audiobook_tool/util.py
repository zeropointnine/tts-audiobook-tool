import copy
import json
import math
import re
import os
import importlib
from pathlib import Path
import platform
import time
from typing import Any, Callable

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.system_support.ansi import Ansi

"""
Various frequently used small util functions, both app-specific and general
Meant to be imported using "*"
"""

# 'Global; variable
_menu_clears_screen: bool = MENU_CLEARS_SCREEN_DEFAULT

def set_menu_clears_screen(value: bool) -> None:
    global _menu_clears_screen
    _menu_clears_screen = value

def printt(s: str="", end=None, dont_reset=False) -> None:
    """
    App-standard way of printing to the console.
    (Doesn't do much extra or different at the moment)
    """
    if not dont_reset:
        s += Ansi.RESET
    print(s, end=end, flush=not bool(end))

def print_feedback(
        message: str,
        end_value: Any = None,
        is_error=False,
        no_preformat=False,
        extra_line=True,
        skip_pause=False
) -> None:
    """
    Should be used for printing feedback after an action is taken (eg, after a setting has been changed),
    and submenu is about to be re-printed.

    :param is_error: if True, prints message in red, and shows an enter prompt
    :param no_pause: if True, doesn't do the typical slight pause
    """
    if not no_preformat:
        message = Ansi.ITALICS + (COL_ERROR if is_error else COL_DIM) + message
    if end_value is not None:
        message = message.strip() + " " + COL_ACCENT + str(end_value)
    printt(message)
    
    if is_error:
        from tts_audiobook_tool import ask
        ask.ask_enter_to_continue()
    else:
        if skip_pause:
            sleep_duration = 0.0
        else:        
            sleep_duration = PRINT_FEEDBACK_PAUSE_NO_CLEAR_SCREEN if _menu_clears_screen else PRINT_FEEDBACK_PAUSE_CLEAR_SCREEN
            if is_error:
                sleep_duration *= 2.0
        time.sleep(sleep_duration)
        
    if extra_line:
        printt()

def make_noun(singular: str, plural: str, quantity: int) -> str:
    return singular if quantity == 1 else plural

def print_init(s: str) -> None:
    """ App style for initializing a thing which may take some time """
    printt(f"{COL_DIM_ITALICS}{s}")
    print()

def print_model_init(model_description: str, extra: str = "") -> None:
    """ 
    Prints model init message in a consistent style 
    """
    s = f"Initializing {model_description} model"
    if extra:
        s += f" {COL_DIM}({extra})"
    s += "..."
    print_init(s)

def make_error_string(e: Exception) -> str:
    """
    Standard way for the app to display exceptions
    """
    return f"{type(e).__name__}: {e}"

def swap_and_delete_file(temp_file_path: str, target_file_path: str) -> str:
    """
    Returns error message on fail, else empty string
    """
    if Path(target_file_path).exists():
        try:
            Path(target_file_path).unlink()
        except Exception as e:
            return f"Couldn't delete original {temp_file_path}, {e}"
    try:
        Path(temp_file_path).rename(target_file_path)
    except Exception as e:
        return f"Couldn't rename {temp_file_path} to {target_file_path}, {e}"
    return ""

def delete_silently(path: str):
    """ Deletes a file and fails silently """
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except Exception as e:
        pass # eat

def is_number(o: Any) -> bool:
    return isinstance(o, int) or isinstance(o, float)

def truncate_pretty(text: str, width: int, middle:bool=True, content_color: str=COL_DEFAULT) -> str:
    """ 
    Truncates string for display. Uses colors.
    Ellipsis in the middle of the string, else at the end
    """
    if len(text) <= width:
        return f"{content_color}{text}"
    if middle:
        width -= 3 # bc triple-dots
        a_len = math.ceil(width / 2)
        b_len = math.floor(width / 2)
        a = text[:a_len]
        b = text[-b_len:]
        return f"{content_color}{a}{COL_DIM}...{content_color}{b}"
    else:
        width -= 3 # bc triple-dots
        a = text[:width]
        return f"{content_color}{a}{COL_DIM}..."

def ellipsize_path_middle(path: str, length: int=60, truncate_file_suffix=False) -> str:
    """ Puts ellipsis before path name if necessary """
    
    if not path:
        return path
    
    p = Path(path)
    if truncate_file_suffix:
        p = p.with_suffix('')

    if len(str(p)) <= length:
        return str(p)

    # Find the separator before path name and construct truncated path
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
    max_suffix_len = length - prefix_len - ellipsis_len
    if len(suffix) > max_suffix_len:
        suffix = suffix[-max_suffix_len:]
    return path[:prefix_len] + "..." + suffix

def ellipsize_path_for_menu(path: str) -> str:
    """ 
    App style for displaying filepath in a menu item
    """
    s = ellipsize(path, 40, from_start=True)
    return s
        
def estimated_wav_seconds(file_path: str) -> float:
    # Assumes 44.1khz, 16 bits, minimal metadata
    num_bytes = 0
    try:
        path = Path(file_path)
        num_bytes = path.stat().st_size
    except:
        return 0
    return num_bytes / (44_100 * 2)

def make_hotkey_string(hotkey: str, color: str="", outer_color: str=Ansi.RESET) -> str:
    if not color:
        color = COL_ACCENT
    return f"{outer_color}[{color}{hotkey}{outer_color}]"

def make_menu_label(
        label: str, 
        value: Any, 
        default: Any=None, 
        value_prefix: str="currently: ", 
        color_code=COL_ACCENT,
        num_decimals=0,
        required_predicate: Callable[[], bool] | None = None
    ) -> str:
    currently = make_currently_string(
        value, value_prefix, default, color_code, num_decimals, required_predicate
    )
    return f"{label} {currently}"

def make_menu_label_optional(label: str) -> str:
    return f"{label} {COL_DIM}(optional{COL_DIM})"

def make_currently_string(
        value: Any, 
        value_prefix: str="currently: ", 
        default: Any=None, 
        color_code=COL_ACCENT,
        num_decimals=0,
        required_predicate: Callable[[], bool] | None = None,
        required_label="required"
    ) -> str:
    """
    Used for presenting the current value for a menu item in a consistent style
    Ex: `(currently: 666)`

    If "required_predicate" is provided and returns True, 
    returns "(required)" to indicate missing value is required.
    """
    if required_predicate and required_predicate():
        return f"{COL_DIM}({COL_ERROR}{required_label}{COL_DIM})"

    value_string = make_parameter_value_string(
        value=value, default=default, num_decimals=num_decimals
    )
    return f"{COL_DIM}({value_prefix}{color_code}{value_string}{COL_DIM})"

@staticmethod
def make_parameter_value_string(
    value: float | int | bool,
    default: float | int | bool,
    num_decimals: int=0
) -> str:

    DEFAULT_LABEL = f" {COL_DIM}default"

    if isinstance(value, bool):
        s = str(value)
        if value == default:
            s += DEFAULT_LABEL
        return s

    if value == -1:
        # Project attributes use -1 to mean "not set explicitly", ie, use default value
        value = default 
    
    if isinstance(value, float):
        if num_decimals == 0:
            s = str(int(value))
        else:
            s = f"{value:.{num_decimals}f}"
    else:
        s = str(value)
    
    if value == default:
        s += DEFAULT_LABEL
    
    return s

def make_gb_string(bytes: int) -> str:
    """ Returns gigabyte string with either one or zero decimal places"""
    gb = bytes / (1024 ** 3)
    gb = int(gb * 10) / 10
    if gb % 1 == 0:
        gb = int(gb)
    return str(gb) + "GB"

def lerp_clamped(
    value: float,
    min_value: float,
    max_value: float,
    mapped_min_value: float,
    mapped_max_value: float,
) -> float:
    """
    Map a value from [min_value, max_value] to [mapped_min_value, mapped_max_value] with clamping.
    """
    normalized = (value - min_value) / (max_value - min_value)
    clamped_normalized = max(0.0, min(1.0, normalized))
    return mapped_min_value + (mapped_max_value - mapped_min_value) * clamped_normalized

def make_file_line_ranges(markers: list[int], num_items: int) -> list[tuple[int, int]]:
    """ 
    Returns ranges with length of markers + 1 (always starts with item with start index 0)
    Assumes `markers` is sorted 
    """

    # TODO: this should be a property in Project

    if not markers:
        return [ (0, num_items - 1) ]

    markers = sorted(markers) # for good measure

    indices = list(markers)
    if indices[0] == 0:
        del indices[0]

    ranges = []

    start = 0
    for index in indices:
        if index < 0 or index >= num_items:
            raise ValueError(f"Out of range: {index}")
        end = index - 1
        range = (start, end)
        ranges.append(range)
        start = index
    range = (start, num_items-1)
    ranges.append(range)

    return ranges

def duration_string(seconds: float, include_tenth: bool=False) -> str:
    """ Returns, eg, 5h0m0s """
    if seconds < 60:
        if include_tenth:
            return f"{seconds:.1f}s"
        else:
            return f"{round(seconds)}s"

    seconds = round(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m{seconds}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes}m{seconds}s"

def time_stamp(seconds: float, with_tenth: bool=True) -> str:
    """ Eg: 05:00:00 """

    tenths = int((seconds - int(seconds)) * 10)
    seconds = int(seconds)

    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60

    hours_string = str(hours).rjust(2, "0")
    minutes_string = str(minutes).rjust(2, "0")
    seconds_string = str(seconds).rjust(2, "0")

    if with_tenth:
        return f"{hours_string}:{minutes_string}:{seconds_string}.{tenths}"
    else:
        return f"{hours_string}:{minutes_string}:{seconds_string}"

def ellipsize(s: str, length: int, from_start:bool = False) -> str:
    if len(s) > length:
        if from_start:
            s = "..." + s[-(length - 3):]
        else:
            s = s[:length - 3] + "..."
    return s

def get_package_dir() -> str | None:
    # Get the current package's root directory
    if not __package__:
        return None
    package = importlib.import_module(__package__)
    if not package.__file__:
        return None
    return os.path.dirname(os.path.abspath(package.__file__))

def make_assets_file_path(file_name: str) -> str:
    """Returns the full path to an asset file in tts_audiobook_tool/assets/."""
    return os.path.join(package_dir, ASSETS_DIR_NAME, file_name)

def make_unique_file_path(file_path: str) -> str:
    """
    Creates a unique file path by adding "-1", "-2", "-3", etc to stem if needed.
    """
    if not os.path.exists(file_path):
        return file_path

    path = Path(file_path)
    suffix = path.suffix
    base_stem = re.sub(r'-\d+$', '', path.stem) # verify
    counter = 1

    while True:
        fn =  base_stem + "-" + str(counter) + suffix
        new_path = path.with_name(fn)
        if not os.path.exists(new_path):
            return str(new_path)
        counter += 1

def is_long_path_enabled():

    if platform.system() != "Windows":
        return True

    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key: # type: ignore
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled") # type: ignore
            return bool(value)
    except FileNotFoundError:
        return False  # Key doesn't exist (older Windows)
    except Exception as e:
        print(f"Error checking registry: {e}")
        return False

def has_gui():
    from tts_audiobook_tool.system_support.platforms import has_gui as system_has_gui

    return system_has_gui()

def open_directory_in_gui(path) -> str:
    from tts_audiobook_tool.system_support.platforms import open_directory

    return open_directory(path)

def is_wsl():
    from tts_audiobook_tool.system_support.platforms import is_wsl as system_is_wsl

    return system_is_wsl()

def clear_input_buffer() -> None:
    from tts_audiobook_tool.system_support.terminal import clear_input_buffer as system_clear_input_buffer

    system_clear_input_buffer()

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

def does_import_test_pass(module_name: str) -> bool:
    """
    Imports module to see if it it exists and appears valid.
    This is more reliable than simply using find_spec() by itself but ofc can induce side-effects.
    """
    # First check if the 'spec' exists at all (is 'side-effect free')
    from importlib.util import find_spec
    if not find_spec(module_name):
        return False

    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_torch_allocated_vram() -> int:
    """
    Returns -1 if no cuda
    Rem: faster-whisper does NOT allocate vram using torch
    """
    import torch
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.memory_allocated()

def print_gen_oom_message(err: str) -> None:
    """
    Print the standard OOM warning followed by the raw error string.
    Used by generate_util.py and real_time_playback_util.py.
    """
    printt(f"{COL_ERROR}{GEN_OOM_ERROR_MESSAGE}")
    printt()
    printt(f"{COL_ERROR}{err}")
    printt()

def is_oom_error_message(error_string: str) -> bool:
    """
    Checks if an error string likely indicates an out-of-memory error.
    Uses pattern matching for common OOM error keywords/phrases from various sources
    (PyTorch, CUDA, system-level, etc.)
    
    Returns True if the error string appears to be an OOM error.
    """
    if not isinstance(error_string, str):
        return False
    
    lower = error_string.lower()
    
    # Common OOM indicators
    oom_patterns = [
        r'out\s+of\s+memory',          # Generic OOM
        r'cuda\s+out\s+of\s+memory',   # CUDA-specific OOM
        r'outofmemory',                 # PyTorch exception name variant
        r'torch\.cuda\.outofmemoryexception',
        r'failed\s+to\s+allocate',     # Memory allocation failure
        r'failed\s+to\s+allocate.*bytes',
        r'failed\s+to\s+malloc',       # malloc failure
        r'failed\s+to\s+malloc.*bytes',
        r'kernel\s+oom',               # Kernel OOM killer
        r'kill.*process',              # OOM killer terminated process
        r'ram\s+full',                 # System RAM full
        r'no\s+space\s+left',          # No space left (on device)
        r'memory\s+exhausted',         # Memory exhausted
        r'memory\s+allocation\s+failed',  # Generic memory alloc failure
    ]
    
    for pattern in oom_patterns:
        if re.search(pattern, lower):
            return True
    return False

def pretty_json_string(payload: dict, ellipsize_at: int=60) -> str:
    """
    Returns pretty json string of dict, ellipsizing long strings
    
    Main use case is to prevent data uri's from flooding the console
    """

    def ellipsize_strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str):
                    value[key] = ellipsize(item, ellipsize_at)
                else:
                    ellipsize_strings(item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str):
                    value[index] = ellipsize(item, ellipsize_at)
                else:
                    ellipsize_strings(item)

    obj = copy.deepcopy(payload)
    ellipsize_strings(obj)
    
    s = json.dumps(obj, indent=2)
    return s
