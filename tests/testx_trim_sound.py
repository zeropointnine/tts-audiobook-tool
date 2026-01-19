import json
import os
from pathlib import Path
from tts_audiobook_tool.app_types import Sound, Strictness, SttConfig, SttVariant
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import printt
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.validation_result import TrimmedResult
from tts_audiobook_tool.whisper_util import WhisperUtil

print(Ansi.CLEAR_SCREEN_AND_SCROLLBACK)

# DIR = "/home/lee/Documents/w/w/_tests/trim"
DIR = "/home/lee/Documents/w/w/_tests/trim_trailing"

Tts.init_model_type()

flacs_and_jsons = []

for file in os.listdir(DIR):
    flac_path = os.path.join(DIR, file)
    if not flac_path.endswith(".flac"):
        continue
    json_path = Path(flac_path).with_suffix(".debug.json")
    if not os.path.exists(json_path):
        continue
    tup = (flac_path, json_path)
    print(tup)
    flacs_and_jsons.append(tup)
printt()

counter = 0        

for flac_path, json_path in flacs_and_jsons:

    sound = SoundFileUtil.load(flac_path)
    assert(isinstance(sound, Sound))
    
    with open(json_path, 'r', encoding='utf-8') as f:
        debug_json = json.load(f)
    prompt = debug_json["prompt"]

    words = WhisperUtil.transcribe_to_words(sound, "en", SttVariant.LARGE_V3, SttConfig.CUDA_FLOAT16)
    assert(isinstance(words, list))

    result = ValidateUtil.validate(sound, prompt, words, "en", Strictness.MODERATE)
    
    if isinstance(result, TrimmedResult):

        printt(f"{COL_DIM}{flac_path}")
        printt("trimmed: " + result.get_ui_message())
        trimmed_path = Path(flac_path).with_suffix(".TRIMMED.flac")
        SoundFileUtil.save_flac(result.sound, str(trimmed_path))

    printt()
