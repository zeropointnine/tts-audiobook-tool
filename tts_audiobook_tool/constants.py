import os
import re

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.app_types import Hint

APP_NAME = "tts-audiobook-tool"

APP_USER_SUBDIR = "tts_audiobook_tool"
APP_TEMP_SUBDIR = "tts_audiobook_tool"
ASSETS_DIR_NAME = "assets"

PROJECT_SOUND_SEGMENTS_SUBDIR = "segments"
PROJECT_CONCAT_SUBDIR = "combined"
PROJECT_JSON_FILE_NAME = "project.json"
PROJECT_TEXT_SEGMENTS_FILE_NAME = "text_segments.json"
PROJECT_TEXT_RAW_FILE_NAME = "text_raw.txt"
PROJECT_CONCAT_TEMP_TEXT_FILE_NAME = "ffmpeg_temp.txt"

PREFS_FILE_NAME = "tts-audiobook-tool-prefs.json"

FFMPEG_COMMAND = "ffmpeg"

STT_TEMP_TRANSCRIBED_WORDS = "temp_words.pkl"

OUTE_DEFAULT_VOICE_JSON_FILE_NAME = "en-female-1-neutral.json"
package_dir = os.path.dirname(os.path.abspath(__file__))
OUTE_DEFAULT_VOICE_JSON_FILE_PATH = os.path.join(package_dir, ASSETS_DIR_NAME, OUTE_DEFAULT_VOICE_JSON_FILE_NAME)

# App should use single sample rate up until outputting final audio
APP_SAMPLE_RATE = 44100

# Samplerate required for whisper input
WHISPER_SAMPLERATE = 16000

# Fish TTS default temperature, taken from their web demo page
DEFAULT_TEMPERATURE_FISH = 0.8

# Default temperature for Higgs V2, taken from their README examples
DEFAULT_TEMPERATURE_HIGGS = 0.3

DEFAULT_SEED = 1000

# App's typical ffmpeg options wrt console output, etc
FFMPEG_TYPICAL_OPTIONS = [
    "-y",  # Overwrite output file if it exists
    "-hide_banner", "-loglevel", "warning",
    "-stats"
]

FFMPEG_ARGUMENTS_OUTPUT_FLAC = [
    "-c:a", "flac",
    "-sample_fmt",  "s16",
    "-frame_size", "4096",
    "-compression_level", "6"
]
FFMPEG_ARGUMENTS_OUTPUT_AAC = [
    "-c:a", "aac",
    "-b:a", f"96k",
    "-movflags", "+faststart", # moves metadata to the front, for streaming, which we want
    '-vn',                    # No video (important when input is mp3 for some reason)
    # Do not use "-sample_fmt s16" here
]

APP_META_FLAC_FIELD = "TTS_AUDIOBOOK_TOOL"
APP_META_MP4_MEAN = "tts-audiobook-tool"
APP_META_MP4_TAG = "audiobook-data"

AAC_SUFFIXES = [".m4a", ".m4b", ".mp4"]

COL_ACCENT = Ansi.hex("ffaa44")
COL_ERROR = Ansi.hex("ff0000")
COL_DIM = Ansi.hex("666666")
COL_INPUT = Ansi.hex("aaaaaa")
COL_OK = Ansi.hex("00ff00")
COL_DEFAULT = Ansi.RESET

PLAYER_URL = "https://zeropointnine.github.io/tts-audiobook-tool/browser_player/"

# App uses this format for file names of audio fragments.
# Example file name: "[00001] [0123456789ABCDEF] [my_voice] [any_other_bracketed_tags] Hello_world.flac"
# Capturing group 1 is segment index - digits enclosed in brackets (eg, "00001")
# Capturing group 2 is hex hash - 16 hex characters enclosed brackets (eg, "0123456789ABCDEF")
# Capturing group 3 is voice label - alphanumeric chars (and underscores) enclosed in brackets
# Rest of string can be anything
pattern = r"\[(\d+)\] \[([0-9A-Fa-f]{16})\] \[(\w+)\] .*"
AUDIO_SEGMENT_FILE_NAME_PATTERN = re.compile(pattern)

# Regex for "[h...]", where "h" is 16 hex characters.
# Captures the hex string (w/o the brackets)
# Eg, "[0123456789ABCDEF]"
# App uses this format for including 64-bit hash values in filenames.
pattern = r'\[([0-9a-fA-F]{16})\]'
HASH_PATTERN = re.compile(pattern)

# ---

HINT_LONG_PATHS = Hint(
    "long_paths",
    "It appears your system does not support long file paths",
    "App relies on pretty long filenames, so be sure to use\nshort directory paths when creating new projects"
)
HINT_OUTE_CONFIG = Hint(
    "oute_config",
    "This appears to be your first time running the application using the Oute TTS model",
    "As a reminder, you should adjust/experiment with the\nsettings in the file config_oute.py for optimal performance."
)
HINT_LINE_BREAKS = Hint(
    "line_breaks",
    "Note:",
    "Line breaks are treated as paragraph delimiters.\nIf your source text uses manual line breaks for word wrapping\n(eg, Project Gutenberg), you will want to reformat it first."
)
HINT_REGEN = Hint(
    "regenerate",
    "Please note",
"""It's oftentimes not possible to get all voice lines to validate,
even after repeated re-generations. Embrace imperfection.

Increasing temperature temporarily can sometimes help."""
)
HINT_REAL_TIME = Hint(
    "real_time",
    "About",
f"""This uses the same quality-control steps as the normal "Generate" workflow,
save for loudness normalization.

To achieve uninterrupted playback, your system must be able to
do the audio inference faster-than-realtime."""
)
HINT_MULTIPLE_MP3S = Hint(
    "multiple_mp3s",
    "Multiple MP3 files?",
    "If you want to combine multiple MP3 files in a directory,\nthis can be done from the Tools/Options menu"
)
HINT_OUTE_LOUD_NORM = Hint(
    "oute_loud_norm",
    "Tip",
    "Oute generations can have considerable variance in loudness.\nConsider using \"stronger.\""
)

HINT_NO_VOICE = Hint(
    "gen_no_voice",
    "No voice clone defined",
    "The TTS model will generate random-sounding voices because no voice sample has been set."
)