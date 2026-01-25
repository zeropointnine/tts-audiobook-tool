from enum import Enum
from functools import cache
from typing import NamedTuple

import numpy as np


class TtsModelInfo(NamedTuple):
    """
    App properties of a supported TTS model
    """
    # TODO: needs to be split between model-properties-proper and program-structure-related maybe

    # Module name to test for that implies the TTS model library exists in the current py env
    module_test: str
    # Supported device types
    torch_devices: list[str]
    # identifier used in file names
    file_tag: str
    # The model's sound output sample rate
    sample_rate: int
    # The app's recommended max-words-per-segment for the model
    max_words_default: int
    # The app's recommended max-words-per-segment range (min, max)
    max_words_reco_range: tuple[int, int]
    # Does the model use a voice clone audio data (eg, Oute does not, wants a json)
    uses_voice_sound_file: bool
    # Does the model require a voice clone sample 
    requires_voice: bool
    # Does the model API require the text transcript of the voice clone sample
    requires_voice_transcript: bool
    # Project field name for "batch size" (must be implemented in Project; empty = no batch support)
    batch_size_project_field: str
    # When True, the "high" setting for strictness is discouraged due to model's poor WER
    strictness_high_discouraged: bool
    # Should semantic trim at end of last word
    # Doing so is generally redundant and risks unintended partial cropping of end of last word
    # due to whisper timing imprecision, but can do more good than harm if model rly likes to 
    # hallucinate past the end of teh prompt (eg, for Chatterbox)
    semantic_trim_last: bool
    # Does the model have a propensity for generating spurious music sounds
    # (ie, should the STT validator check for music)
    hallucinates_music: bool
    # Forces lowercase on prompts that start out all-caps (see `un_all_caps_prompt()`)
    un_all_caps: bool
    # The requirements.txt file that should be used to install the virtual environment for the given tts model
    requirements_file_name: str
    # ui-related strings and values
    ui: dict
    # List of string replace pairs 
    # Primarily used for punctuation marks that models might either disregard or trigger them in other ways
    substitutions: list[ tuple[str, str] ]

    @property
    def can_batch(self) -> bool:
        return bool(self.batch_size_project_field)

class TtsModelInfos(Enum):
    """
    Enumerates `TtsModelInfo` instances for all supported TTS models
    """

    # Placeholder
    NONE = TtsModelInfo(
        module_test="",
        file_tag="",
        torch_devices = [],
        sample_rate=0,
        max_words_default=0,
        max_words_reco_range=(0, 0),
        uses_voice_sound_file=False,
        requires_voice=False,
        requires_voice_transcript=False,
        batch_size_project_field="",
        strictness_high_discouraged=True,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=False,
        requirements_file_name="",
        ui = {},
        substitutions=[]
    )

    OUTE = TtsModelInfo(
        module_test="outetts",
        file_tag="oute",
        torch_devices = [], # not applicable
        sample_rate=44100,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        uses_voice_sound_file=False,
        requires_voice=True,
        requires_voice_transcript=False,
        batch_size_project_field="",
        strictness_high_discouraged=True,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=False, # TODO: check this
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
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        uses_voice_sound_file=True,
        requires_voice=False,
        requires_voice_transcript=False,
        batch_size_project_field="",
        strictness_high_discouraged=True,
        semantic_trim_last=True,
        hallucinates_music=False,
        un_all_caps=True,
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
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=44100,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        uses_voice_sound_file=True,
        requires_voice=False,
        requires_voice_transcript=True,
        batch_size_project_field="",
        strictness_high_discouraged=False,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=True, # Does well with all caps, but still worse than normal case
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
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        uses_voice_sound_file=True,
        requires_voice=False,
        requires_voice_transcript=True,
        batch_size_project_field="",
        strictness_high_discouraged=False,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=False, # TODO: did very ltd (cpu-bound) test only
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
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        uses_voice_sound_file=True,
        requires_voice=False,
        requires_voice_transcript=False,
        batch_size_project_field="vibevoice_batch_size",
        strictness_high_discouraged=True,
        semantic_trim_last=False,
        hallucinates_music=True,
        un_all_caps=True,
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
        ]
    )

    INDEXTTS2 = TtsModelInfo(
        module_test="indextts",
        file_tag="indextts2",
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=22050,
        max_words_default=40,
        max_words_reco_range=(40, 60),
        uses_voice_sound_file=True,
        requires_voice=True,
        requires_voice_transcript=False,
        batch_size_project_field="",
        strictness_high_discouraged=False,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=False,
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
        ]
    )

    GLM = TtsModelInfo(
        module_test="glm_tts",
        file_tag="glm",
        torch_devices = ["cuda"], # cuda-only atm
        sample_rate=24000,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        uses_voice_sound_file=True,
        requires_voice=True,
        requires_voice_transcript=True,
        batch_size_project_field="",
        strictness_high_discouraged=False,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=False,
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
        ]
    )

    MIRA = TtsModelInfo(
        module_test="mira",
        file_tag="mira",
        torch_devices = [], # does not take in a device as a parameters
        sample_rate=48000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        uses_voice_sound_file=True,
        requires_voice=True,
        requires_voice_transcript=False,
        batch_size_project_field="mira_batch_size",
        strictness_high_discouraged=False,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=True, # falls down badly with all caps phrases
        requirements_file_name="requirements-mira.txt",
        ui = {
            "proper_name": "MiraTTS",
            "short_name": "Mira",
            "voice_path_console": "Enter voice clone audio clip file path (recommended up to 15s): ",
            "voice_path_requestor": "Select voice clone audio clip (recommended up to 15s)"
        },
        substitutions=[
            # semicolon doesn't create caesura, but neither does comma reliably
            # em-dash and space-en-dash-space seems okay
            # "caesura punctuation" in general seems unpredictable so there's no use replacing characters
        ]
    )

    QWEN3TTS = TtsModelInfo(
        module_test="qwen_tts",
        file_tag="qwen3",
        torch_devices = ["cuda", "mps", "cpu"],
        sample_rate=24000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        uses_voice_sound_file=True,
        requires_voice=False, # actually depends on model_type
        requires_voice_transcript=True,
        batch_size_project_field="qwen3_batch_size",
        strictness_high_discouraged=False,
        semantic_trim_last=False,
        hallucinates_music=False,
        un_all_caps=True, # is only slightly more error-prone when all-caps
        requirements_file_name="requirements-qwen3tts.txt",
        ui = {
            "proper_name": "Qwen3-TTS",
            "short_name": "Qwen3-TTS",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip"
        },
        substitutions=[
            # Does rly well w/ various punctuation
        ]
    )

    @staticmethod
    def recommended_range_string(info: TtsModelInfo) -> str:
        if info.max_words_reco_range[1] == info.max_words_reco_range[0]:
            return f"up to {info.max_words_reco_range[1]}"
        else:
            return f"{info.max_words_reco_range[0]}-{info.max_words_reco_range[1]}"

    @staticmethod
    @cache
    def all_file_tags() -> set[str]:
         return { item.value.file_tag for item in TtsModelInfos }
