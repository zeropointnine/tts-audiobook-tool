from __future__ import annotations
from enum import Enum
from functools import cache
from typing import NamedTuple

from tts_audiobook_tool.app_types import DeviceType


class TtsModelSpec(NamedTuple):
    """
    Hardcoded properties of a supported TTS model
    """

    # identifier used for serialization
    id: str
    # Whether the model is backed by an external backend rather than local inference
    is_sgl_omni: bool
    # Substring to use for simple model matching against SGL-Omni model name (empty = not applicable)
    server_model_id_substring: str
    # Module name, or "dist:<package>" / "dist:<package>==<version>", to test for that implies the TTS model library exists in the current py env
    local_module_test: str
    # Supported torch device types for local inference
    local_torch_devices: list[DeviceType]
    # identifier used in file names
    file_tag: str
    # The model's sound output sample rate
    sample_rate: int
    # The app's recommended max-words-per-segment for the model
    max_words_default: int
    # The app's recommended max-words-per-segment range (min, max)
    max_words_reco_range: tuple[int, int]
    
    # Project attribute of voice clone file name (when applicable)
    voice_target_attr: str
    # Does the model require a voice clone sample to generate audio
    requires_voice: bool
    # Project attribute of voice clone transcript (when applicable; empty = none)
    voice_transcript_attr: str
    # Additional project attributes for model-specific saved files
    extra_file_attrs: list[str]
    # Project field name for "batch size" (must be implemented in Project; empty = no batch support)
    
    batch_size_attr: str
    # Whether the model supports streaming chunk callbacks
    can_stream: bool
    # Should semantic trim at end of last word
    # Doing so is generally redundant and risks unintended partial cropping of end of last word
    # due to whisper timing imprecision, but can do more good than harm if model rly likes to 
    # hallucinate past the end of teh prompt (eg, for Chatterbox)
    semantic_trim_last: bool
    # Does the model require FFmpeg shared libraries (dll/so/dylib), not just the ffmpeg executable
    # In practice, this is usually because the model depends on TorchCodec
    requires_ffmpeg_libs: bool
    # Forces lowercase on prompts that start out all-caps (see `un_all_caps_prompt()`).
    # Should be set for models that perform poorly on all-caps text
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
        return bool(self.batch_size_attr)

class TtsModelType(Enum):
    """
    Enumerates all supported TTS models using `TtsModelSpec` instances
    """

    # Placeholder model type when no other models are detected
    NONE = TtsModelSpec(
        id="none",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="",
        local_torch_devices = [],
        file_tag="",
        sample_rate=0,
        max_words_default=0,
        max_words_reco_range=(0, 0),
        voice_target_attr="",
        requires_voice=False,
        voice_transcript_attr="",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="",
        ui = {
            "proper_name": "None",
            "short_name": "None",
            "voice_path_console": "",
            "voice_path_requestor": "",
            "project_links": ["https://www.example.com"]
        },
        substitutions=[]
    )

    CHATTERBOX = TtsModelSpec(
        id="chatterbox",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="chatterbox",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="chatterbox",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        voice_target_attr="chatterbox_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=True,
        requires_ffmpeg_libs=False,
        un_all_caps=True,
        requirements_file_name="requirements-chatterbox.txt",
        ui = {
            "proper_name": "Chatterbox TTS",
            "short_name": "Chatterbox",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/resemble-ai/chatterbox", "https://huggingface.co/ResembleAI"]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ")
        ]
    )

    FISH_S1 = TtsModelSpec(
        id="fish_s1",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="dist:fish-speech==0.1.0",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="s1-mini",
        sample_rate=44_100,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="fish_s1_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="fish_s1_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=True, # Does well with all caps, but still worse than normal case
        requirements_file_name="requirements-fish-s1.txt",
        ui = {
            "proper_name": "Fish S1-mini",
            "short_name": "S1-mini",
            "voice_path_console": "Enter voice clone audio clip file path (up to 10s): ",
            "voice_path_requestor": "Select voice clone audio clip (up to 10s)",
            "project_links": ["https://github.com/fishaudio/fish-speech", "https://huggingface.co/fishaudio/s1-mini"]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ") # em dash does not reliably induce caesura
        ]
    )

    FISH_S2 = TtsModelSpec(
        id="fish_s2",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="dist:fish-speech==2.0.0",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="s2-pro",
        sample_rate=44_100,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="fish_s2_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="fish_s2_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-fish-s2.txt",
        ui = {
            "proper_name": "Fish S2-Pro",
            "short_name": "S2-Pro",
            "voice_path_console": "Enter voice clone audio clip file path (10-30s): ",
            "voice_path_requestor": "Select voice clone audio clip (10-30s)",
            "project_links": ["https://github.com/fishaudio/fish-speech", "https://huggingface.co/fishaudio/s2-pro"]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ") # em dash does not reliably induce caesura
        ]
    )

    FISH_S2_SERVER = TtsModelSpec(
        id="server_fish_s2",
        is_sgl_omni=True,
        server_model_id_substring="fish", # b/c sgl omni only supports one type of fish model which is v2
        local_module_test="",
        local_torch_devices = [],
        file_tag="s2-pro",
        sample_rate=44_100,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="fish_s2_server_voice_target",
        requires_voice=False, 
        voice_transcript_attr="fish_s2_server_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="fish_s2_server_concurrent_requests",
        can_stream=True,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-fish-s2.txt",
        ui = {
            "proper_name": "Fish S2-Pro",
            "short_name": "S2-Pro",
            "voice_path_console": "",
            "voice_path_requestor": "",
            "project_links": [
                "https://github.com/fishaudio/fish-speech", 
                "https://huggingface.co/fishaudio/s2-pro", 
                "https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html"
            ]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ") # em dash does not reliably induce caesura
        ]
    )

    GLM = TtsModelSpec(
        id="glm",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="glm_tts",
        local_torch_devices = [DeviceType.CUDA], # cuda-only atm
        file_tag="glm",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        voice_target_attr="glm_voice_file_name",
        requires_voice=True,
        voice_transcript_attr="glm_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-glm.txt",
        ui = {
            "proper_name": "GLM-TTS",
            "short_name": "GLM",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/zai-org/GLM-TTS", "https://huggingface.co/zai-org/GLM-TTS"]
        },
        substitutions=[
            (";", ","), # semicolon generates random syllable
            ("\u2014", ", "), ("\u2500", ", "), # em-dash doesn't create caesura
            (" \u2013 ", ", ") # space-en-dash-space doesn't create caesura
        ]
    )

    HIGGS_V2 = TtsModelSpec(
        id="higgs_v2",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="boson_multimodal",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="higgs_v2",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        voice_target_attr="higgs_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="higgs_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-higgs-v2.txt",
        ui = {
            "proper_name": "Higgs Audio V2",
            "short_name": "Higgs v2",
            "voice_path_console": "Enter voice clone audio clip file path (~15 seconds recommended): ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/boson-ai/higgs-audio", "https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base"]
        },
        substitutions=[ (
            "\u2014", ", "), ("\u2500", ", ")
        ]
    )

    HIGGS_V3_SERVER = TtsModelSpec(
        id="server_higgs_v3",
        is_sgl_omni=True,
        server_model_id_substring="higgs", # b/c sgl omni only supports one type of higgs model, which is v3
        local_module_test="",
        local_torch_devices = [],
        file_tag="higgs_v3",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="higgs_v3_voice_target",
        requires_voice=False,
        voice_transcript_attr="higgs_v3_voice_transcript", 
        extra_file_attrs=[],
        batch_size_attr="higgs_v3_batch_size",
        can_stream=True,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-sgl-omni.txt",
        ui = {
            "proper_name": "Higgs Audio V3",
            "short_name": "Higgs v3",
            "voice_path_console": "", # not applicable
            "voice_path_requestor": "", # not applicable
            "project_links": [
                "https://huggingface.co/bosonai/higgs-audio-v3-tts-4b", 
                "https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html"
            ]
        },
        substitutions=[ (
            "\u2014", ", "), ("\u2500", ", ")
        ]
    )

    INDEXTTS2 = TtsModelSpec(
        id="indextts2",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="indextts",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="indextts2",
        sample_rate=22_050,
        max_words_default=40,
        max_words_reco_range=(40, 60),
        voice_target_attr="indextts2_voice_file_name",
        requires_voice=True,
        voice_transcript_attr="",
        extra_file_attrs=["indextts2_emo_voice_file_name"],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-indextts2.txt",
        ui = {
            "proper_name": "IndexTTS2",
            "short_name": "IndexTTS2",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/index-tts/index-tts", "https://huggingface.co/IndexTeam/IndexTTS-2"]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", "), # em-dash oftentimes doesn't create caesura
            ("\u2013", ", ") # en-dash oftentimes generates random syllable
        ]
    )

    MIRA = TtsModelSpec(
        id="mira",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="mira",
        local_torch_devices = [], # does not take in a device as a parameters
        file_tag="mira",
        sample_rate=48_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="mira_voice_file_name",
        requires_voice=True,
        voice_transcript_attr="",
        extra_file_attrs=[],
        batch_size_attr="mira_batch_size",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=True, # falls down badly with all caps phrases
        requirements_file_name="requirements-mira.txt",
        ui = {
            "proper_name": "MiraTTS",
            "short_name": "Mira",
            "voice_path_console": "Enter voice clone audio clip file path (recommended up to 15s): ",
            "voice_path_requestor": "Select voice clone audio clip (recommended up to 15s)",
            "project_links": ["https://github.com/ysharma3501/MiraTTS", "https://huggingface.co/YatharthS/MiraTTS"]
        },
        substitutions=[
            # semicolon doesn't create caesura, but neither does comma reliably
            # em-dash and space-en-dash-space seems okay
            # "caesura punctuation" in general seems unpredictable so there's no use replacing characters
        ]
    )

    MOSS = TtsModelSpec(
        id="moss",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="dist:moss-tts",
        local_torch_devices=[DeviceType.CUDA, DeviceType.CPU],
        file_tag="moss",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="moss_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="moss_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="moss_batch_size",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=True,
        un_all_caps=False,
        requirements_file_name="requirements-moss.txt",
        ui={
            "proper_name": "MOSS-TTS",
            "short_name": "MOSS",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/OpenMOSS/MOSS-TTS", "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5"]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ")
        ]
    )

    MOSS_SERVER = TtsModelSpec(
        id="server_moss",
        is_sgl_omni=True,
        server_model_id_substring="moss",
        local_module_test="",
        local_torch_devices=[],
        file_tag="moss",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="moss_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="moss_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="moss_batch_size",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=True,
        un_all_caps=False,
        requirements_file_name="requirements-sgl-omni.txt",
        ui={
            "proper_name": "MOSS-TTS",
            "short_name": "MOSS",
            "voice_path_console": "",
            "voice_path_requestor": "",
            "project_links": [
                "https://github.com/OpenMOSS/MOSS-TTS", 
                "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5", 
                "https://sgl-project.github.io/sglang-omni/cookbook/moss_tts.html"
            ]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", ")
        ]
    )

    OMNIVOICE = TtsModelSpec(
        id="omnivoice",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="omnivoice",
        local_torch_devices=[DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="omnivoice",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80), 
        voice_target_attr="omnivoice_voice_file_name",
        requires_voice=False, # supports Voice Design and Auto Voice without ref_audio
        voice_transcript_attr="omnivoice_voice_transcript", # Preempts OmniVoice from using internal Whisper instance for transcription
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=True, # slightly more error-prone when all-caps
        requirements_file_name="requirements-omnivoice.txt",
        ui={
            "proper_name": "OmniVoice",
            "short_name": "OmniVoice",
            "voice_path_console": "Enter voice clone audio clip file path (3-10s recommended, up to 15s): ",
            "voice_path_requestor": "Select voice clone audio clip (3-10s recommended, up to 15s)",
            "project_links": ["https://github.com/k2-fsa/OmniVoice", "https://huggingface.co/k2-fsa/OmniVoice"]
        },
        substitutions=[
            # Generally handles caesura-related punctuation decently and with natural variations in duration
        ]
    )

    OUTE = TtsModelSpec(
        id="oute",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="outetts",
        local_torch_devices = [], # not applicable
        file_tag="oute",
        sample_rate=44_100,
        max_words_default=40,
        max_words_reco_range=(40, 40),
        voice_target_attr="oute_voice_json", # rem, special case, is not a sound file
        requires_voice=True,
        voice_transcript_attr="",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False, # TODO: check this
        requirements_file_name="requirements-oute.txt",
        ui = {
            "proper_name": "Oute TTS",
            "short_name": "Oute",
            "voice_path_console": "Enter voice clone audio clip file path (up to 15s): ",
            "voice_path_requestor": "Select voice clone audio clip (up to 15s)",
            "project_links": ["https://github.com/edwko/OuteTTS", "https://huggingface.co/OuteAI"]
        },
        substitutions=[
            # fyi u2500 = "box drawing light horizontal". have seen it in the wild used as an em-dash.
            ("\u2014", ", "), ("\u2500", ", ")
        ]
    )

    POCKET = TtsModelSpec(
        id="pocket",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="pocket_tts",
        local_torch_devices=[DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="pocket",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="pocket_voice_file_name",
        requires_voice=True,
        voice_transcript_attr="",
        extra_file_attrs=[],
        batch_size_attr="",
        can_stream=True,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=False,
        requirements_file_name="requirements-pocket.txt",
        ui={
            "proper_name": "Pocket TTS",
            "short_name": "Pocket",
            "voice_path_console": "Enter voice clone audio clip file path (5-10s, up to 15s): ",
            "voice_path_requestor": "Select voice clone audio clip (5-10s, up to 15s)",
            "project_links": ["https://github.com/kyutai-labs/pocket-tts", "https://huggingface.co/kyutai/pocket-tts"],
            "opt_in_url": "https://github.com/kyutai-labs/pocket-tts" # special case
        },
        substitutions=[
            # Generally handles caesura-related punctuation decently and with natural variations in duration
        ]
    )

    QWEN3TTS = TtsModelSpec(
        id="qwen3tts",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="qwen_tts",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="qwen3",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="qwen3_voice_file_name",
        requires_voice=True, # this applies to 'base' model type only
        voice_transcript_attr="qwen3_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="qwen3_batch_size",
        can_stream=False,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=True, # is only slightly more error-prone when all-caps
        requirements_file_name="requirements-qwen3tts.txt",
        ui = {
            "proper_name": "Qwen3-TTS",
            "short_name": "Qwen3-TTS",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/QwenLM/Qwen3-TTS", "https://huggingface.co/collections/Qwen/qwen3-tts"]
        },
        substitutions=[] # Does well w/ various punctuation
    )

    QWEN3TTS_SERVER = TtsModelSpec(
        id="server_qwen3tts",
        is_sgl_omni=True,
        server_model_id_substring="qwen",
        local_module_test="",
        local_torch_devices=[],
        file_tag="qwen3",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="qwen3_voice_file_name",
        requires_voice=True, # Note, this diverges from the local version
        voice_transcript_attr="qwen3_voice_transcript",
        extra_file_attrs=[],
        batch_size_attr="qwen3_server_concurrent_requests",
        
        # SGL-Omni API formally supports streaming, and the app is hooked up to handle it,
        # but it DOES NOT STREAM in practice (chunks come in a burst all at the end), 2026-06.
        can_stream=False,
        
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=True,
        requirements_file_name="requirements-sgl-omni.txt",
        ui={
            "proper_name": "Qwen3-TTS",
            "short_name": "Qwen3-TTS",
            "voice_path_console": "",
            "voice_path_requestor": "",
            "project_links": [
                "https://github.com/QwenLM/Qwen3-TTS",
                "https://huggingface.co/collections/Qwen/qwen3-tts",
                "https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html"
            ]
        },
        substitutions=[] # Does well w/ various punctuation
    )

    VIBEVOICE = TtsModelSpec(
        id="vibevoice",
        is_sgl_omni=False,
        server_model_id_substring="",
        local_module_test="vibevoice",
        local_torch_devices = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU],
        file_tag="vibevoice",
        sample_rate=24_000,
        max_words_default=40,
        max_words_reco_range=(40, 80),
        voice_target_attr="vibevoice_voice_file_name",
        requires_voice=False,
        voice_transcript_attr="",
        extra_file_attrs=[],
        batch_size_attr="vibevoice_batch_size",
        can_stream=True,
        semantic_trim_last=False,
        requires_ffmpeg_libs=False,
        un_all_caps=True,
        requirements_file_name="requirements-vibevoice.txt",
        ui = {
            "proper_name": "VibeVoice",
            "short_name": "VibeVoice",
            "voice_path_console": "Enter voice clone audio clip file path: ",
            "voice_path_requestor": "Select voice clone audio clip",
            "project_links": ["https://github.com/vibevoice-community/VibeVoice"]
        },
        substitutions=[
            ("\u2014", ", "), ("\u2500", ", "), (";", ","), # em dash and semicolon oftentimes don't create caesuras
            ("\u2019", "'"), # fancy apostrophe causes rest of word to not be spoken
            ("…", ","), ("...", ",") # ellipsis can wreck gen badly
        ]
    )

    # ---

    @staticmethod
    def get_by_id(id: str) -> TtsModelType:
        for item in TtsModelType:
            if item.value.id == id:
                return item
        return TtsModelType.NONE

    @staticmethod
    def recommended_range_string(info: TtsModelSpec) -> str:
        if info.max_words_reco_range[1] == info.max_words_reco_range[0]:
            return f"up to {info.max_words_reco_range[1]}"
        else:
            return f"{info.max_words_reco_range[0]}-{info.max_words_reco_range[1]}"

    @staticmethod
    @cache
    def all_file_tags() -> set[str]:
         return { item.value.file_tag for item in TtsModelType }
    
    @staticmethod
    @cache
    def get_sgl_omni_items() -> list[TtsModelType]:
        result = []
        for item in TtsModelType:
            if item.value.is_sgl_omni:
                result.append(item)
        return result

    @staticmethod
    def find_tts_type_using_sgl_omni_model_id(model_id: str) -> TtsModelType | None:
        """
        Chooses TtsModelType member using the model id returned by 
        the SGL-Omni models endpoint (typically an hf repo id),
        using simple substring comparison.
        """
        if not model_id:
            return None
        
        model_id = model_id.lower().strip()
        
        for item in TtsModelType.get_sgl_omni_items():
            substring = item.value.server_model_id_substring.lower()                
            if substring and substring in model_id:
                return item
            
        return None
