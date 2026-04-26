"""
Constants - config-like values
TODO: consider making this a config file proper
"""

import os

DEV = os.getenv("TTS_AUDIOBOOK_TOOL_DEV", "").lower() in ("true", "1", "yes") and True

PROJECT_DEFAULT_LANGUAGE = "en"
PROJECT_DEFAULT_SECTION_SOUND_EFFECT = False
PROJECT_DEFAULT_REALTIME_SAVE = False
PROJECT_DEFAULT_LIMIT_SILENCE_GAPS = False

PROJECT_MAX_RETRIES_DEFAULT = 1
PROJECT_MAX_RETRIES_MIN = 1
PROJECT_MAX_RETRIES_MAX = 5
PROJECT_BATCH_SIZE_DEFAULT = 1
PROJECT_BATCH_SIZE_MAX = 99

PREFS_DEFAULT_PLAY_ON_GENERATE = False

# Max words per text chunk, applied to the source text in "STT flow"
MAX_WORDS_PER_SEGMENT_STT = 40

PAUSE_DURATION_WORD = 0.1
PAUSE_DURATION_PHRASE = 0.5
PAUSE_DURATION_SENTENCE = 0.9
PAUSE_DURATION_PARAGRAPH = 1.2
PAUSE_DURATION_SECTION = 2.0
PAUSE_DURATION_UNDEFINED = 1.0

LIMIT_SILENCE_GAPS_DURATION = 1.0

REAL_TIME_BUFFER_MAX_SECONDS = 60 * 5

# Use 0.0 to skip probability filtering
VOICE_CLONE_TRANSCRIBE_MIN_PROBABILITY = 0.75 

DEFAULT_LLM_CONVERSATION_SYSTEM_PROMPT = (
    "You are having a voice conversation with the user. "
    "The user is speaking into the computer microphone, transcribed using speech-to-text. " 
    "Be aware that minor errors may occur in the transcription process. "
    "Your responses are converted into speech using text-to-speech. "
    "Be conversational and brief. "
    "Respond in plain, natural language that sounds good when spoken aloud. Avoid emojis, decorative or nonverbal punctuation. Use only simple ASCII-friendly characters, prefer the regular dash `-`, and avoid em dashes, en dashes, or unusual symbols.`"
)

