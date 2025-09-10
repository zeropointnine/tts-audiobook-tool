from enum import Enum
from functools import cache
from typing import NamedTuple


class TtsInfo(NamedTuple):
    """
    App properties of a supported TTS model
    """
    # bucket of ui-related strings and values
    ui: dict
    # Module name to test for that implies the TTS model library exists in the current py env
    module_test: str
    # identifier used in file names
    file_tag: str
    # the model's output sample rate
    sample_rate: int
    # Should semantic trim return end time stamp if is last word
    # Doing so is generally redundant and risks unintended partial cropping of end of last word,
    # but can be useful for chopping off hallucinated noises past last word (Chatterbox)
    semantic_trim_last: bool
    # Model does not respect em-dashes in terms of 'prosody', so replace with some other puncutation which it will
    em_dash_replace: str

    # The requirements.txt file that should be used to install the virtual environment for the tts model
    requirements_file_name: str

# ---

OUTE_SPECS = TtsInfo(
    ui = {
        "proper_name": "Oute TTS",
        "short_name": "Oute",
        "voice_path_console": "Enter voice clone audio clip file path (up to 15s) (wav, flac, or mp3): ", # no m4a
        "voice_path_requestor": "Select voice clone audio clip (up to 15s) (wav, flac, or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".mp3"]
    },
    module_test="outetts",
    file_tag="oute",
    sample_rate=44100,
    semantic_trim_last=False,
    em_dash_replace=", ", # helps maybe
    requirements_file_name="requirements-oute.txt"
)
CHATTERBOX_SPECS = TtsInfo(
    ui = {
        "proper_name": "Chatterbox TTS",
        "short_name": "Chatterbox",
        "voice_path_console": "Enter voice clone audio clip (wav, flac, m4a or mp3): ",
        "voice_path_requestor": "Select voice clone audio clip (wav, flac, m4a or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".m4a", ".mp3"]
    },
    module_test="chatterbox",
    file_tag="chatterbox",
    sample_rate=24000,
    semantic_trim_last=True,
    em_dash_replace=": ", # helps
    requirements_file_name="requirements-chatterbox.txt"
)
FISH_SPECS = TtsInfo(
    ui = {
        "proper_name": "Fish S1-mini",
        "short_name": "S1-mini",
        "voice_path_console": "Enter voice clone audio clip file path (up to 10s) (wav, flac, or mp3): ", # no m4a
        "voice_path_requestor": "Select voice clone audio clip (up to 10s) (wav, flac, or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".mp3"]
    },
    module_test="fish_speech",
    file_tag="s1-mini",
    sample_rate=44100,
    semantic_trim_last=False,
    em_dash_replace="", # TODO fish does need this; choose punctuation for it
    requirements_file_name="requirements-fish.txt"
)
HIGGS_SPECS = TtsInfo(
    ui = {
        "proper_name": "Higgs Audio V2",
        "short_name": "higgs",
        "voice_path_console": "Enter voice clone audio clip file path (wav, flac, or mp3): ", # TODO m4a? TODO duration?
        "voice_path_requestor": "Select voice clone audio clip (wav, flac, or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".mp3"]
    },
    module_test="boson_multimodal",
    file_tag="higgs",
    sample_rate=24000,
    semantic_trim_last=False, # TODO
    em_dash_replace="", # TODO
    requirements_file_name="requirements-higgs.txt"
)
VIBEVOICE_SPECS = TtsInfo(
    ui = {
        "proper_name": "VibeVoice",
        "short_name": "Vibe Voice",
        "voice_path_console": "Enter voice clone audio clip file path (wav, flac, or mp3): ",
        "voice_path_requestor": "Select voice clone audio clip (wav, flac, or mp3)",
        "voice_path_suffixes": [".wav", ".flac", ".mp3"]
    },
    module_test="vibevoice",
    file_tag="vibevoice",
    sample_rate=24000,
    semantic_trim_last=False, # TODO
    em_dash_replace="", # TODO
    requirements_file_name="requirements-vibevoice.txt"
)

class TtsType(Enum):

    NONE = TtsInfo({}, "", "none", 0, False, "", "")
    OUTE = OUTE_SPECS
    CHATTERBOX = CHATTERBOX_SPECS
    FISH = FISH_SPECS
    HIGGS = HIGGS_SPECS
    VIBEVOICE = VIBEVOICE_SPECS

    @staticmethod
    @cache
    def all_file_tags() -> set[str]:
         return { item.value.file_tag for item in TtsType }

