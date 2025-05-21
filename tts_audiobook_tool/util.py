import re
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.ansi import Ansi

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
    length = len(s)
    printt(f"{COL_ACCENT}{s}")
    printt("-" * length)

def ask(message: str="", lower: bool=True, extra_line: bool=True) -> str:
    """
    App-standard way of getting user input.
    Prints extra line after the input by default.
    """
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

def ask_confirm(message: str="") -> bool:
    if not message:
        message = f"Press {make_hotkey_string("Y")} to confirm: "
    inp = ask_hotkey(message)
    return inp == "y"

def strip_ansi_codes(s: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s)

def make_random_hex_string(num_hex_chars: int=32) -> str:
    return f"{random.getrandbits(num_hex_chars * 4):0{num_hex_chars}x}"

def is_ffmpeg_available() -> bool:
    """Check if 'ffmpeg' is installed and accessible in the system PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def encode_to_flac(wav_path: str, flac_path: str) -> bool:

    try:
        # Construct the FFmpeg command with proper escaping for filenames
        cmd = [
            'ffmpeg',
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            '-i', wav_path,
            '-c:a', 'flac',
            '-compression_level', '5',
            flac_path
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True

    except subprocess.CalledProcessError as e:
        printt(str(e.stderr), "error")
        return False
    except Exception as e:
        printt(str(e), "error")
        return False

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
    except:
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

