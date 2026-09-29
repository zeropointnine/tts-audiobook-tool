from __future__ import annotations
from enum import Enum
from typing import NamedTuple

from tts_audiobook_tool.app_types import DeviceType


class TtsBackendKind(Enum):
    """
    How a TTS model variant is executed or served.

    LOCAL: the variant is run by its model library inside the current
    virtual environment (one venv per model, see `.agents/venv-models.md`).
    SGL_OMNI: the variant is served over HTTP by an external SGL-Omni
    (SGLang) server.

    The `TtsModelType.NONE` placeholder is not a real model and has no
    backend; its `TtsModelSpec.backend_kind` is `None`.
    """

    LOCAL = "local"
    SGL_OMNI = "sgl_omni"


class TtsModelSpec(NamedTuple):
    """
    Application metadata for a supported TTS model (built-in or configured)
    """

    # identifier used for serialization
    id: str
    # The kind of backend that executes/serves this variant;
    # None on the NONE placeholder, which is not a real model and has no backend
    backend_kind: TtsBackendKind | None
    # Substring to use for simple model matching against SGL-Omni model name (empty = not applicable)
    sgl_omni_model_id_substring: str
    # Module name, or "dist:<package>" / "dist:<package>==<version>", to test for that implies the TTS model library exists in the current py env
    local_module_test: str
    # Supported torch device types for local inference
    local_torch_devices: list[DeviceType]
    # identifier used in file names
    file_tag: str
    # The model's native/default sound output sample rate
    default_output_sample_rate: int

    # Does the model require a voice clone sample to generate audio
    requires_voice: bool
    # Whether the model supports streaming chunk callbacks
    can_stream: bool
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
    # Case-sensitive substrings identifying worker console lines to be filtered out of output history
    output_filters: list[str]
    # List of string replace pairs
    # Primarily used for punctuation marks that models might either disregard or trigger them in other ways
    substitutions: list[ tuple[str, str] ]

