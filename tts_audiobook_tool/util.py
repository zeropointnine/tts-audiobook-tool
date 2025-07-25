import json
import re
import os
import random
import importlib
from datetime import datetime
from pathlib import Path
import platform
import subprocess
from typing import Any

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.l import L


def printt(s: str="") -> None:
    """
    App-standard way of printing to the console.
    (Doesn't do anything extra or different at the moment)
    """
    s += Ansi.RESET
    print(s)

def print_heading(s: str, dont_clear: bool=False) -> None:
    """ """
    if MENU_CLEARS_SCREEN and not dont_clear:
        os.system('cls' if os.name == 'nt' else 'clear')

    length = get_string_printable_len(s)
    printt("-" * length)
    printt(f"{COL_ACCENT}{s}")
    printt("-" * length)

def ask(message: str="", lower: bool=True, extra_line: bool=True) -> str:
    """
    App-standard way of getting user input.
    Prints extra line after the input by default.
    """

    if not DEV:
        clear_input_buffer()

    message = f"{message}{COL_INPUT}"
    try:
        inp = input(message).strip()
    except (ValueError, EOFError) as e:
        return ""
    if lower:
        inp = inp.lower()
    print(Ansi.RESET, end="")
    if extra_line:
        printt()
    return inp

def ask_hotkey(message: str="", lower: bool=True, extra_line: bool=True) -> str:
    inp = ask(message, lower, extra_line)
    if inp:
        inp = inp[0]
    return inp

def ask_file_path(
        console_message: str,
        requestor_title: str,
        filetypes: list[tuple[str, str]] = [],
        initialdir: str=""
) -> str:
    try:
        from tkinter import filedialog
        printt(console_message)
        result = filedialog.askopenfilename(title=requestor_title, filetypes=filetypes, initialdir=initialdir)
        printt(result)
        printt()
        return result
    except Exception as e:
        pass
    return ask_path_input(console_message)

def ask_dir_path(
        console_message: str,
        ui_title: str,
        initialdir: str = "",
        mustexist: bool = True,
) -> str:
    try:
        from tkinter import filedialog
        printt(console_message)
        result = filedialog.askdirectory(title=ui_title, initialdir=initialdir, mustexist=mustexist) # fyi, mustexist doesn't rly do anything on Windows
        printt(result)
        printt()
        return result
    except Exception as e:
        pass
    return ask_path_input(console_message)

def ask_path_input(message: str="") -> str:
    """
    Get file/directory path, strip outer quotes
    Could potentially open standard file requestor here
    """
    printt(message)
    inp = ask("")
    return strip_quotes_from_ends(inp)

def ask_confirm(message: str="") -> bool:
    if not message:
        message = f"Press {make_hotkey_string("Y")} to confirm: "
    inp = ask_hotkey(message)
    return inp == "y"

def ask_continue(message_prefix: str="") -> None:
    message = "Press enter: "
    if message_prefix:
        message = f"{message_prefix} {message}"
    ask(message)

def ask_error(message_prefix: str) -> None:
    message = f"{COL_ERROR}{message_prefix}"
    message += "\nPress enter: "
    ask(message)

def strip_quotes_from_ends(s: str) -> str:
    if len(s) >= 2:
        first = s[0]
        last = s[-1]
        if (first == "'" and last == "'") or (first == "\"" and last == "\""):
            s = s[1:-1]
    return s

def strip_ansi_codes(s: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s)

def make_random_hex_string(num_hex_chars: int=32) -> str:
    return f"{random.getrandbits(num_hex_chars * 4):0{num_hex_chars}x}"

def make_sibling_random_file_path(source_file_path: str, new_suffix: str="") -> str:
    """
    When no new_suffix, uses source_file_path's suffix
    """
    source_path = Path(source_file_path)
    parent_dir = source_path.parent
    new_suffix = new_suffix or source_path.suffix
    new_file_name = make_random_hex_string() + new_suffix
    new_path = os.path.join(parent_dir, new_file_name)
    return new_path

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

def timestamp_string() -> str:
    current_time = datetime.now()
    return current_time.strftime("%y%m%d_%H%M%S")

def estimated_wav_seconds(file_path: str) -> float:
    # Assumes 44.1khz, 16 bits, minimal metadata
    num_bytes = 0
    try:
        path = Path(file_path)
        num_bytes = path.stat().st_size
    except:
        return 0
    return num_bytes / (44_100 * 2)

def delete_silently(path: str):
    """ Deletes a file and fails silently """
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except Exception as e:
        L.w(f"Couldn't delete temp file {path} {e}")
        pass # eat

def make_hotkey_string(hotkey: str, color: str=COL_ACCENT) -> str:
    return f"[{color}{hotkey}{Ansi.RESET}]"

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

def massage_for_text_comparison(s: str) -> str:
    # Massages text so that source text and transcribed text can be compared
    s = s.lower().strip()
    # First replace fancy apost with normal apost
    s = s.replace("’", "'") #
    # Replace all non-alpha-numerics with space, except for apost that is inside a word
    s = re.sub(r"[^a-zA-Z0-9'’]|(?<![a-zA-Z])['’]|['’](?![a-zA-Z])", ' ', s)
    # Strip white space from the ends
    s = re.sub(r' +', ' ', s)
    s = s.strip(' ')
    # Standardize the spelling of numbers from 1-20
    # (More could be done past 20, but let's not)
    s = substitute_smol_numbers(s)
    return s

def substitute_smol_numbers(text) -> str:
    """Replace standalone numbers 1-20 with their written equivalents (in lowercase)."""
    number_map = {
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
        '10': 'ten',
        '11': 'eleven',
        '12': 'twelve',
        '13': 'thirteen',
        '14': 'fourteen',
        '15': 'fifteen',
        '16': 'sixteen',
        '17': 'seventeen',
        '18': 'eighteen',
        '19': 'nineteen',
        '20': 'twenty'
    }
    # Use regex to find standalone numbers (surrounded by word boundaries)
    pattern = r'\b(?:' + '|'.join(number_map.keys()) + r')\b'
    # Replace each found number with its word equivalent
    result = re.sub(
        pattern,
        lambda match: number_map[match.group()],
        text
    )
    return result


def sanitize_for_filename(filename: str) -> str:
    """
    Replaces all non-alpha-numeric characters with underscores,
    replaces consecutive underscores with a single underscore,
    and removes leading/trailing underscores.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', filename)
    collapsed = re.sub(r'_+', '_', sanitized)
    stripped = collapsed.strip('_')
    return stripped


def make_section_ranges(section_dividers: list[int], num_items: int) -> list[tuple[int, int]]:
    """ Assumes `section_dividers` is sorted """

    # TODO: this should be a property in Project

    if not section_dividers:
        return [ (0, num_items - 1) ]

    indices = list(section_dividers)
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
    """ eg, 5h0m0s """
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
    """ 05:00:00 """

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

def ellipsize(s: str, length: int) -> str:
    if len(s) > length:
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

def get_unique_file_path(file_path: str) -> str:
    """
    Generates a unique file path by adding incrementing numbers to stem if needed.
    """
    file_path = re.sub(r'-\d+$', '', file_path)
    path = Path(file_path)
    counter = 1
    while path.exists():
        path = path.with_stem(f"{path.stem}-{str(counter)}")
        counter = 1
    return str(path)

def is_long_path_enabled():

    if platform.system() != "Windows":
        return True

    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(value)
    except FileNotFoundError:
        return False  # Key doesn't exist (older Windows)
    except Exception as e:
        print(f"Error checking registry: {e}")
        return False

def has_gui():
    """Check if a GUI shell exists or whatever"""
    s = platform.system()
    if s == "Linux":
        return "DISPLAY" in os.environ  # X11 GUI environment check # This also returns true when using WSL
    elif s == "Darwin":  # macOS (assumes GUI is available)
        return True
    elif s == "Windows":
        return True  # Assume GUI is available on Windows
    else:
        return False  # Unknown system

def open_directory_in_gui(path) -> str:
    """
    Open the directory in the OS's default file explorer.
    Returns error string on fail
    """
    if not os.path.isdir(path):
        return "Directory doesn't exist"
    if not has_gui():
        return "No recognized GUI environment detected"

    if is_wsl():
        return "Unsupported for WSL"

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and others
            subprocess.run(["xdg-open", path])
    except Exception as e:
        return f"Failed to open directory: {e}"
    return ""

def is_wsl():
    if not platform.system() == "Linux":
        return False
    # Combine checks for robustness
    checks = [
        "microsoft" in platform.uname().release.lower(),
        "WSL_DISTRO_NAME" in os.environ,
        os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
    ]
    return any(checks)

def clear_input_buffer() -> None:
    """ Use before "input()" to prevent buffered keystrokes from being registered """
    import sys
    try:
        # Windows
        import msvcrt
        while msvcrt.kbhit(): # type: ignore
            msvcrt.getch()
    except ImportError:
        # Linux/macos
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH) # type: ignore

def get_string_printable_len(string: str) -> int:
    """
    Returns the length of the string, filtering out non-printable characters like ANSI codes.
    """
    # ANSI escape code pattern: \x1b\[[0-?]*[ -/]*[@-~]
    # This pattern covers most common ANSI SGR (Select Graphic Rendition) codes.
    # It matches:
    # \x1b or \033 (ESC)
    # \[ (opening bracket)
    # [0-?]* (zero or more characters in the range 0x30-0x3F, typically numbers and semicolons)
    # [ -/]* (zero or more intermediate characters in the range 0x20-0x2F)
    # [@-~] (final character in the range 0x40-0x7E, which indicates the end of the sequence)
    ansi_escape_pattern = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

    # Remove ANSI escape codes
    clean_string = ansi_escape_pattern.sub('', string)

    return len(clean_string)

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
