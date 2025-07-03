"""
"""

# Max words per text chunk, applied to the source text.
MAX_WORDS_PER_SEGMENT = 40
# Max words per text chunk, applied to the source text for "STT mode"
MAX_WORDS_PER_SEGMENT_STT = 40

MENU_CLEARS_SCREEN = True

PREFS_DEFAULT_PLAY_ON_GENERATE = False
PREFS_DEFULT_SHOULD_NORMALIZE = True

# Offset for whisper word end timestamp being consistently too early
# The amount varies too much, is usually around 0.15, but is always too early.
# This is obviously very model-specific
WHISPER_END_TIME_OFFSET = 0.40

WHISPER_START_TIME_OFFSET = -0.1

PAUSE_DURATION_SENTENCE = 1.0
PAUSE_DURATION_PARAGRAPH = 1.2
PAUSE_DURATION_INSIDE_SENTENCE = 0.5
PAUSE_DURATION_UNDEFINED = 1.0

REAL_TIME_BUFFER_MAX_SECONDS = 60 * 5

