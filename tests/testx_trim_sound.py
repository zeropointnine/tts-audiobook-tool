from tts_audiobook_tool.app_types import Sound, Strictness, SttVariant, SttConfig, TrimmableResult
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.whisper_util import WhisperUtil


path = "/home/lee/Documents/w/w/_tests/[00015] [01766245711554] [raw].flac"
sound = SoundFileUtil.load(path)
assert(isinstance(sound, Sound))

source_text = "The man smiled."

words = WhisperUtil.transcribe_to_words(sound=sound,  stt_variant=SttVariant.LARGE_V3, stt_config=SttConfig.CUDA_FLOAT16 , language_code="en")
assert(isinstance(words, list))

trimmable_result = ValidateUtil.validate_item(sound, source_text, words, "en", strictness=Strictness.MODERATE)
if not isinstance(trimmable_result, TrimmableResult):
    print("sound not trimmable, choose smth else")
    exit()

# SoundUtil.trim(sound, )


... # continue this

# Speech-to-text validation (0.21s): Fixed - Excess words detected but reference text exists as substring. Will remove 0 to 0.80 and 2.00 to end 2.00
