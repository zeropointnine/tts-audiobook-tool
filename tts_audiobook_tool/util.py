import re
import os
from pathlib import Path
import random
import subprocess
import re
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

def ask(message: str="", lower: bool=True, extra_line: bool=True) -> str:
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

