"""
Constants - config-like values
TODO: externalize to config file
"""

import os

DEV = os.getenv("TTS_AUDIOBOOK_TOOL_DEV", "").lower() in ("true", "1", "yes") and True
DEV_SAVE_INTERMEDIATE_FILES = DEV and True
if DEV:
    print("\n### DEV ###")

# Max words per text chunk, applied to the source text.
MAX_WORDS_PER_SEGMENT = 40
# Max words per text chunk, applied to the source text for "STT mode"
MAX_WORDS_PER_SEGMENT_STT = 40

MENU_CLEARS_SCREEN = False

PREFS_DEFAULT_PLAY_ON_GENERATE = False
PREFS_DEFAULT_NORMALIZATION_LEVEL = "default"

# Offset for whisper word end timestamp being consistently too early
# The amount varies a lot, usually around 0.15, but is always too early.
# This is ofc very stt-model-specific
WHISPER_END_TIME_OFFSET = 0.25

WHISPER_START_TIME_OFFSET = -0.1

PAUSE_DURATION_PARAGRAPH = 1.2
PAUSE_DURATION_SENTENCE = 0.9
PAUSE_DURATION_PHRASE = 0.5
PAUSE_DURATION_WORD = 0.2
PAUSE_DURATION_UNDEFINED = 1.0

REAL_TIME_BUFFER_MAX_SECONDS = 60 * 5

VOICE_TRANSCRIBE_MIN_PROBABILITY = 0.75 # Use 0.0 to skip probability filtering
