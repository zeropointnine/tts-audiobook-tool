import re
import os
import random
import importlib
from datetime import datetime
from pathlib import Path
import platform
import subprocess

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.l import L

def printt(s: str="", type: str="") -> None:
    if type == "disabled":
        s = strip_ansi_codes(s)
        s = Ansi.hex("666666") + s
    elif type == "error":
        s = strip_ansi_codes(s)
        s = Ansi.hex("ff0000") + s
    s += Ansi.RESET
    print(s)

    if type == "error":
        ask("\nPress enter: ")
        printt()

def print_heading(s: str) -> None:
    """ """
    length = len(s) # TODO need to filter out control codes :/
    printt(f"{COL_ACCENT}{s}")
    printt("-" * length)

def ask(message: str="", lower: bool=True, extra_line: bool=True) -> str:
    """
    App-standard way of getting user input.
    Prints extra line after the input by default.
    """

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

def ask_path(message: str="") -> str:
    """
    Get file/directory path, strip outer quotes
    Could potentially open standard file requestor here
    """
    inp = ask(message, lower=False, extra_line=True)
    return strip_quotes_from_ends(inp)

def strip_quotes_from_ends(s: str) -> str:
    if len(s) >= 2:
        first = s[0]
        last = s[-1]
        if (first == "'" and last == "'") or (first == "\"" and last == "\""):
            s = s[1:-1]
    return s

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

def delete_temp_file(path: str):
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

def massage_for_text_comparison(s: str) -> str:
    # Massages text so that source text can be reliably compared to transcribed text
    s = s.lower().strip()
    # First replace fancy apost with normal apost
    s = s.replace("’", "'") #
    # Replace all non-alpha-numerics with space, except for apost that is inside a word
    s = re.sub(r"[^a-zA-Z0-9'’]|(?<![a-zA-Z])['’]|['’](?![a-zA-Z])", ' ', s)
    s = re.sub(r' +', ' ', s)
    s = s.strip(' ')
    return s

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

def has_gui():
    """Check if the environment supports opening a GUI file explorer etc"""
    s = platform.system()
    if s == "Linux":
        return "DISPLAY" in os.environ  # X11 GUI environment check
    elif s == "Darwin":  # macOS (assumes GUI is available)
        return True
    elif s == "Windows":
        return True  # Assume GUI is available on Windows
    else:
        return False  # Unknown system

def open_directory_gui(path) -> str:
    """
    Open the directory in the OS's default file explorer.
    Returns error string on fail
    """
    if not os.path.isdir(path):
        return "Directory doesn't exist"

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)  # Works on Windows
        elif system == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and others
            subprocess.run(["xdg-open", path])
    except Exception as e:
        return f"Failed to open directory: {e}"
    return ""

def clear_input_buffer() -> None:
    """ Use before "input()" to prevent buffered keystrokes from being registered """
    import sys
    try:
        import msvcrt # only exists if windows
        def clear_input_buffer():
            while msvcrt.kbhit(): # type: ignore
                msvcrt.getch()
    except ImportError:
        import termios  # Only exists if linux/macos
        def clear_input_buffer():
            termios.tcflush(sys.stdin, termios.TCIOFLUSH) # type: ignore
