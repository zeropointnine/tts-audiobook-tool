import os
import re

from tts_audiobook_tool.system_support.ansi import Ansi

package_dir = os.path.dirname(os.path.abspath(__file__))

APP_NAME = "tts-audiobook-tool"
APP_URL = "https://github.com/zeropointnine/tts-audiobook-tool"

APP_USER_SUBDIR = "tts_audiobook_tool"
APP_TEMP_SUBDIR = "tts_audiobook_tool"
ASSETS_DIR_NAME = "assets"
CHROME_USER_DATA_DIR_NAME = "chromium-user-data"

PROJECT_SOUND_SEGMENTS_SUBDIR = "segments"
PROJECT_CONCAT_SUBDIR = "combined"
PROJECT_REALTIME_SUBDIR = "realtime"
PROJECT_JSON_FILE_NAME = "project.json"
PROJECT_TEXT_FILE_NAME = "project_text.json"
PROJECT_TEXT_SEGMENTS_FILE_NAME = PROJECT_TEXT_FILE_NAME
PROJECT_TEXT_RAW_FILE_NAME = "project_text_raw.txt"
PROJECT_TEXT_EPUB_FILE_NAME = "project_text.epub"
PROJECT_CONCAT_TEMP_TEXT_FILE_NAME = "ffmpeg_temp.txt"

FFMPEG_COMMAND = "ffmpeg"

STT_TEMP_TRANSCRIBED_WORDS = "temp_words.pkl"
VALIDATION_UNSUPPORTED_LANGUAGES = ["zh", "ja", "ko"]

# App's samplerate for final outputs (post-processed sound segments, sound output stream, etc).
APP_SAMPLE_RATE = 48000

# Samplerate required for whisper audio input
WHISPER_SAMPLERATE = 16000

MAX_WORDS_PER_SEGMENT_DEFAULT = 40
MAX_WORDS_PER_SEGMENT_MIN = 20
MAX_WORDS_PER_SEGMENT_MAX = 80

TOP_P_MIN_DEFAULT = 0.01
TOP_P_MAX_DEFAULT = 1.0

TOP_K_MIN_DEFAULT = 1
TOP_K_MAX_DEFAULT = 100

REPETITION_PENALTY_MIN_DEFAULT = 1.0
REPETITION_PENALTY_MAX_DEFAULT = 2.0

SEED_MAX = 2**32 - 1

CTRANSLATE_REQUIRED_CUDNN_VERSION = 91002

OUTE_DEFAULT_VOICE_JSON_FILE_NAME = "en-female-1-neutral.json"
OUTE_DEFAULT_VOICE_JSON_FILE_PATH = os.path.join(package_dir, ASSETS_DIR_NAME, OUTE_DEFAULT_VOICE_JSON_FILE_NAME)

MENU_CLEARS_SCREEN_DEFAULT = True

# Value used for normalization after any sound transform post-processing steps (eg, after high-shelf EQ)
NORMALIZATION_HEADROOM_DB = 1.0

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
AAC_BITRATES = ["64k", "96k"]
AAC_BITRATE_DEFAULT = "96k"

FFMPEG_ARGUMENTS_OUTPUT_AAC_TEMPLATE = [
    "-c:a", "aac",
    "-b:a", "%1",
    "-movflags", "+faststart", # moves metadata to the front, for streaming, which we want
    '-vn',                    # No video (important when input is mp3 for some reason)
    # Do not use "-sample_fmt s16" here
]

def make_ffmpeg_arguments_output_aac(bitrate: str=AAC_BITRATE_DEFAULT) -> list[str]:
    if bitrate not in AAC_BITRATES:
        bitrate = AAC_BITRATE_DEFAULT
    return [item.replace("%1", bitrate) for item in FFMPEG_ARGUMENTS_OUTPUT_AAC_TEMPLATE]

# Backwards-compatible default AAC args
FFMPEG_ARGUMENTS_OUTPUT_AAC = make_ffmpeg_arguments_output_aac()

APP_META_FLAC_FIELD = "TTS_AUDIOBOOK_TOOL"
APP_META_MP4_MEAN = "tts-audiobook-tool"
APP_META_MP4_TAG = "audiobook-data"
ABR_VERSION = 2
PROJECT_SPEC_VERSION = 2

AAC_SUFFIXES = [".m4a", ".m4b", ".mp4"]

COL_ACCENT = Ansi.hex("ffaa44")
COL_ERROR = Ansi.hex("ff0000")
COL_DIM = Ansi.hex("888888")
COL_MEDIUM = Ansi.hex("cccccc")
COL_INPUT = Ansi.hex("aaaaaa")
COL_OK = Ansi.hex("00ff00")
COL_DEFAULT = Ansi.RESET # default text color being that of the terminal; we're assuming this is probably a light color
COL_DIM_ITALICS = COL_DIM + Ansi.ITALICS

GEN_OOM_ERROR_MESSAGE = "Likely out-of-memory error.\nStopping generation to prevent further failed attempts:"

PLAYER_URL = "https://zeropointnine.github.io/tts-audiobook-tool/browser_player/"

SECTION_SOUND_EFFECT_PATH = os.path.join(package_dir, ASSETS_DIR_NAME, "page-turn-a.wav")

FILE_REQUESTOR_SOUND_TYPES = [('Sound files', '*.wav *.flac *.mp3,*.aac,*.m4a,*.ogg'), ('All files', '*.*')]

# App uses this format for file names of audio fragments.
# Example file name: "[00001] [0123456789ABCDEF] [my_voice] [any_other_bracketed_tags] Hello_world.flac"
# Capturing group 1 is segment index - digits enclosed in brackets (eg, "00001")
# Capturing group 2 is hex hash - 16 hex characters enclosed brackets (eg, "0123456789ABCDEF")
# Capturing group 3 is voice label - alphanumeric chars (and underscores) enclosed in brackets
# Rest of string can be anything
pattern = r"\[(\d+)\] \[([0-9A-Fa-f]{16})\] \[(\w+)\] .*"
SOUND_SEGMENT_FILE_NAME_PATTERN = re.compile(pattern)

# Regex for "[h...]", where "h" is 16 hex characters.
# Captures the hex string (w/o the brackets)
# Eg, "[0123456789ABCDEF]"
# App uses this format for including 64-bit hash values in filenames.
pattern = r'\[([0-9a-fA-F]{16})\]'
HASH_PATTERN = re.compile(pattern)

VOICE_ADVANCED_SUPERLABEL = "Advanced:"

OPT_IN_INSTRUCTIONS = (
    "[1] Visit %1\n"
    "    and authorize access using a logged-in Hugging Face account.\n"
    "[2] Run `hf auth login` and enter valid Hugging Face access token.\n"
    "[3] Restart the app"
)
