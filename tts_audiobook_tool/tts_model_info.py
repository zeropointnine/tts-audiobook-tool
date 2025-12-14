from enum import Enum
from functools import cache
from typing import Any, NamedTuple

import numpy as np


class TtsModelInfo(NamedTuple):
    """
    App properties of a supported TTS model
    """
    # Module name to test for that implies the TTS model library exists in the current py env
    module_test: str
    # Supported device types
    torch_devices: list[str]
    # identifier used in file names
    file_tag: str
    # The model's output dtype
    dtype: np.dtype
    # The model's output sample rate
    sample_rate: int
    # The key used for Prefs json dict to store the max-words value
    max_words_prefs_key: str
    # The model's recommended max-words-per-segment
    max_words_default: int
    # Should semantic trim return end time stamp if is last word
    # Doing so is generally redundant and risks unintended partial cropping of end of last word,
    # but can be useful for chopping off hallucinated noises past last word (eg, for Chatterbox)
    semantic_trim_last: bool
    # The requirements.txt file that should be used to install the virtual environment for the tts model
    requirements_file_name: str
    # ui-related strings and values
    ui: dict
    # List of string replace pairs (used for "untrained" punctuation primarily)
    substitutions: list[ tuple[str, str] ]

class TtsModelInfos(Enum):
    """
    Enumerates `TtsModelInfo` instances for all TTS models supported by the app
    """

    NONE = TtsModelInfo("", [], "none", np.dtype("float32"), 0, "", 0, False, "", {}, []) # placeholder

    OUTE = TtsModelInfo(
        module_test="outetts",
        file_tag="oute",
        dtype=np.dtype("float32"),
        torch_devices = [], # not applicable
        sample_rate=44100,
        max_words_prefs_key="max_words_oute",
        max_words_default=40,
        semantic_trim_last=False,
        requirements_file_name="requirements-oute.txt",
        ui = {
            "proper_name": "Oute TTS",
            "short_name": "Oute",
            "voice_path_console": "Enter voice clone audio clip file path (up to 15s): ",
            "voice_path_requestor": "Select voice clone audio clip (up to 15s)"
        },
        substitutions=[
            # fyi u2500 = "box drawing light horizontal". have seen it in the wild used as an em-dash.
            ("\u2014", ", "), ("\u2500", ", ")
        ]
    )

    CHATTERBOX = TtsModelInfo(
        module_test="chatterbox",
        file_tag="chatterbox",
        dtype=np.dtype("float32"),
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_prefs_key="max_words_chatterbox",
        max_words_default=40,
        semantic_trim_last=True,
        requirements_file_name="requirements-chatterbox.txt",
        ui = {
            "proper_name": "Chatterbox TTS",
            "short_name": "Chatterbox",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip"
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ")
        ]
    )

    FISH = TtsModelInfo(
        module_test="fish_speech",
        file_tag="s1-mini",
        dtype=np.dtype("float32"),
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=44100,
        max_words_prefs_key="max_words_fish",
        max_words_default=40,
        semantic_trim_last=False,
        requirements_file_name="requirements-fish.txt",
        ui = {
            "proper_name": "Fish S1-mini",
            "short_name": "S1-mini",
            "voice_path_console": "Enter voice clone audio clip file path (up to 10s): ",
            "voice_path_requestor": "Select voice clone audio clip (up to 10s)"
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ") # em dash does not reliably induce caesura
        ]
    )

    HIGGS = TtsModelInfo(
        module_test="boson_multimodal",
        file_tag="higgs",
        dtype=np.dtype("float32"),
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_prefs_key="max_words_higgs",
        max_words_default=40,
        semantic_trim_last=False,
        requirements_file_name="requirements-higgs.txt",
        ui = {
            "proper_name": "Higgs Audio V2",
            "short_name": "higgs",
            "voice_path_console": "Enter voice clone audio clip file path (~15 seconds recommended): ",
            "voice_path_requestor": "Select voice clone audio clip"
        },
        substitutions=[ (
            "\u2014", ", "), ("\u2500", ", ")
        ]
    )

    VIBEVOICE = TtsModelInfo(
        module_test="vibevoice",
        file_tag="vibevoice",
        dtype=np.dtype("float32"),
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_prefs_key="max_words_higgs",
        max_words_default=40,
        semantic_trim_last=False,
        requirements_file_name="requirements-vibevoice.txt",
        ui = {
            "proper_name": "VibeVoice",
            "short_name": "Vibe Voice",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip"
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", "), (";", ","), # em dash and semicolon oftentimes don't create caesuras
            ("\u2019", "'"), # fancy apostrophe causes rest of word to not be spoken
            ("…", ","), ("...", ",") # ellipsis can wreck gen badly
        ],
    )

    INDEXTTS2 = TtsModelInfo(
        module_test="indextts",
        file_tag="indextts2",
        dtype=np.dtype("float32"),
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=22050,
        max_words_prefs_key="max_words_indextts2",
        max_words_default=40,
        semantic_trim_last=False,
        requirements_file_name="requirements-indextts2.txt",
        ui = {
            "proper_name": "IndexTTS2",
            "short_name": "IndexTTS2",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip"
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", "), # em-dash oftentimes doesn't create caesura
            ("\u2013", ", ") # en-dash oftentimes generates random syllable
        ],
    )

    GLM = TtsModelInfo(
        module_test="glm_tts",
        file_tag="glm",
        dtype=np.dtype("float32"),
        torch_devices = ["cuda"], # cuda-only atm
        sample_rate=24000,
        max_words_prefs_key="max_words_indextts2",
        max_words_default=40,
        semantic_trim_last=False,
        requirements_file_name="requirements-glm.txt",
        ui = {
            "proper_name": "GLM-TTS",
            "short_name": "GLM",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip"
        },
        substitutions=[
            (";", ","), # semicolon generates random syllable
            ("\u2014", ", "), ("\u2500", ", "), # em-dash doesn't create caesura
            (" \u2013 ", ", ") # space-en-dash-space doesn't create caesura
        ],
    )

    @staticmethod
    @cache
    def all_file_tags() -> set[str]:
         return { item.value.file_tag for item in TtsModelInfos }
