import os
import re

from tts_audiobook_tool.ansi import Ansi

APP_NAME = "tts-audiobook-tool"

# Max words per text chunk, applied to the source text.
MAX_WORDS_PER_SEGMENT = 40
# Max words per text chunk, applied to the source text for "STT mode"
MAX_WORDS_PER_SEGMENT_STT = 40

SENTENCE_CONTINUATION_MAX_DURATION = 0.5
PARAGRAPH_SILENCE_MIN_DURATION = 0.75

MENU_CLEARS_SCREEN = False

SETTINGS_FILE_NAME = "tts-audiobook-tool-settings.json"

PREFS_FILE_NAME = "tts-audiobook-tool-prefs.json"
PREFS_PLAY_ON_GENERATE_DEFAULT = False
PREFS_SHOULD_NORMALIZE = True
PREFS_OPTIMIZE_SS_DEFAULT = True

PROJECT_JSON_FILE_NAME = "project.json"
PROJECT_TEXT_SEGMENTS_FILE_NAME = "text_segments.json"
PROJECT_TEXT_RAW_FILE_NAME = "text_raw.txt"
PROJECT_CONCAT_TEMP_TEXT_FILE_NAME = "ffmpeg_temp.txt"
FFMPEG_COMMAND = "ffmpeg"

APP_TEMP_SUBDIR = "tts_audiobook_tool"

STT_TEMP_TRANSCRIBED_WORDS = "temp_words.pkl"

AUDIO_SEGMENTS_SUBDIR = "segments"
CONCAT_SUBDIR = "combined"

ASSETS_DIR_NAME = "assets"
DEFAULT_VOICE_JSON_FILE_NAME = "en-female-1-neutral.json"
package_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE_JSON_FILE_PATH = os.path.join(package_dir, ASSETS_DIR_NAME, DEFAULT_VOICE_JSON_FILE_NAME)


# Note use single value app-wide, 44k matching Oute's output sample rate
# Also, un-complicates concatenation of mixed model outputs
FLAC_OUTPUT_SAMPLE_RATE = 44100

# App's standardized ffmpeg arguments for outputting FLAC files
FLAC_OUTPUT_FFMPEG_ARGUMENTS = [
    "-c:a", "flac",
    "-sample_fmt",  "s16",
    "-frame_size", "4096",
    "-ar", f"{FLAC_OUTPUT_SAMPLE_RATE}",
    "-compression_level", "6"
]

APP_META_FLAC_FIELD = "TTS_AUDIOBOOK_TOOL"
APP_META_MP4_MEAN = "tts-audiobook-tool"
APP_META_MP4_TAG = "audiobook-data"

MP4_SUFFIXES = [".mp4", ".m4a", ".m4b"]

COL_ACCENT = Ansi.hex("ffaa44")
COL_ERROR = Ansi.hex("ff0000")
COL_DIM = Ansi.hex("666666")
COL_INPUT = Ansi.hex("aaaaaa")
COL_OK = Ansi.hex("00ff00")
COL_DEFAULT = Ansi.RESET

PLAYER_URL = "https://zeropointnine.github.io/tts-audiobook-tool/browser_player/"

# App uses this format for file names of audio fragments.
# Example file name: "[00001] [0123456789ABCDEF] [my_voice] [pass] Hello_world.flac"
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

