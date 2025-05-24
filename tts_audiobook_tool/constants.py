import os
import re

from tts_audiobook_tool.ansi import Ansi

APP_NAME = "tts-audiobook-tool"

DEFAULT_TEMPERATURE = 0.5

# Different voices may have a varying rates of delivery.
# Oute recommends 30s max but can handle longer generations.
# In practice though, when it approaches around 25-30s,
# it appears to internally 'segment' the audio itself,
# which can lead to significant discontinuities/pauses.
MAX_WORDS_PER_SEGMENT = 40

MENU_CLEARS_SCREEN = True

SETTINGS_FILE_NAME = "tts-audiobook-tool-settings.json"
PREFS_FILE_NAME = "tts-audiobook-tool-prefs.json"
PROJECT_VOICE_FILE_NAME = "voice.json"
PROJECT_SETTINGS_FILE_NAME = "project.json"
PROJECT_RAW_TEXT_FILE_NAME = "text raw.txt"
PROJECT_FFMPEG_CONCAT_TEMP_FILE_NAME = "ffmpeg_temp.txt"
FFMPEG_COMMAND = "ffmpeg"

AUDIO_SEGMENTS_SUBDIR = "segments"
CONCAT_SUBDIR = "combined"

ASSETS_DIR_NAME = "assets"
DEFAULT_VOICE_FILE_NAME = "en-female-1-neutral.json"
package_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE_FILE_PATH = os.path.join(package_dir, ASSETS_DIR_NAME, DEFAULT_VOICE_FILE_NAME)

APP_META_FLAC_FIELD = "TTS_AUDIOBOOK_TOOL"
APP_META_MP4_MEAN = "tts-audiobook-tool"
APP_META_MP4_TAG = "audiobook-data"


COL_ACCENT = Ansi.hex("ffaa44")
COL_ERROR = Ansi.hex("ff0000")
COL_DIM = Ansi.hex("666666")
COL_INPUT = Ansi.hex("aaaaaa")
COL_DEFAULT = Ansi.RESET

PLAYER_URL = "https://zeropointnine.github.io/tts-audiobook-tool/browser_player/"

# Regex for "[h...]", where "h" is 16 hex characters.
# Captures the hex string (w/o the brackets)
# Eg, "[0123456789ABCDEF]"
# App uses this format for including 64-bit hash values in filenames.
pattern = r'\[([0-9a-fA-F]{16})\]'
HASH_PATTERN = re.compile(pattern)

# Regex
# Eg, "[some_voice_id] [00001] [0123456789ABCDEF] some voice line text here" ... with optional 4th bracketed tag, "[pass]"
# Capturing group 1 is any number of digits enclosed in second set of brackets (eg, "00001")
# Capturing group 2 is a string of exactly 16 hex characters enclosed in third set of brackets (eg, "0123456789ABCDEF")
# App uses this format for file names of audio fragments.
pattern = r"\[.*?\] \[(\d+)\] \[([0-9A-Fa-f]{16})\] .*"
AUDIO_SEGMENT_FILE_NAME_PATTERN = re.compile(pattern)
