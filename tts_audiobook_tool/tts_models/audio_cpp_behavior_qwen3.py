"""Qwen3-TTS checkpoint type and ICL-only Base cloning.

Discovery identifies the family, not the checkpoint type. Known download filenames
can suggest the type on explicit selection; the user can still override it. This
choice controls client capabilities and payloads, never the server checkpoint.
The server remains the authority on actual checkpoint compatibility.
"""
from __future__ import annotations

import ntpath
from typing import ClassVar

from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues


# Exact GGUF package filenames from audio.cpp model_specs/qwen3_tts.json.
# Deliberately exclude directory names, inferred quantizations and loose matches.
_QWEN3_DOWNLOAD_MODEL_TYPES = {
    "qwen3-tts-12hz-0.6b-base-q8_0.gguf": "base",
    "qwen3-tts-12hz-0.6b-base-bf16.gguf": "base",
    "qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf": "base",
    "qwen3-tts-12hz-1.7b-base-bf16.gguf": "base",
    "qwen3-tts-12hz-1.7b-base-orig.gguf": "base",
    "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf": "custom_voice",
    "qwen3-tts-12hz-1.7b-customvoice-bf16.gguf": "custom_voice",
    "qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf": "voice_design",
    "qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf": "voice_design",
}


def infer_qwen3_model_type_from_path(path: object) -> str | None:
    """Recognize download basenames without depending on the server's path OS."""
    if not isinstance(path, str):
        return None
    return _QWEN3_DOWNLOAD_MODEL_TYPES.get(ntpath.basename(path))


class Qwen3Behavior(AudioCppModelBehavior):
    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {
        "model_type": "choice",
    }

    def can_voice_clone(self, values: ParameterValues) -> bool:
        return values["model_type"] == "base"

    def can_voice_presets(self, values: ParameterValues) -> bool:
        return values["model_type"] == "custom_voice"

    def requires_voice_preset(self, values: ParameterValues) -> bool:
        return self.can_voice_presets(values)

    def uses_reference_transcript(self, values: ParameterValues) -> bool:
        return self.can_voice_clone(values)

    def uses_voice_instructions(self, values: ParameterValues) -> bool:
        return values["model_type"] in ("custom_voice", "voice_design")

    def expects_voice_instructions(self, values: ParameterValues) -> bool:
        return values["model_type"] == "voice_design"

    def resolve_language(self, language: str) -> str:
        # Full names also work on older audio.cpp servers that reject ISO codes.
        languages = {
            "zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean",
            "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
            "es": "spanish", "it": "italian",
        }
        return language if language in languages.values() else languages.get(language, "auto")

    def get_ignored_parameters(self, values: ParameterValues) -> frozenset[str]:
        return frozenset({"model_type"})

    def adjust_payload(self, payload: dict, values: ParameterValues) -> None:
        # Qwen3 samples each frame in two stages: the main talker (codebook 0) reads
        # temperature/top_k/top_p, while the subtalker (remaining codebooks) has its own
        # subtalker_* controls. Mirror them, as the local backend does, so a user setting
        # governs the whole frame rather than only codebook 0.
        for key in ("temperature", "top_k", "top_p"):
            if key in payload:
                payload.setdefault("options", {})[f"subtalker_{key}"] = payload[key]
        if self.can_voice_clone(values):
            # Match local Base policy and override any server default for x-vector mode.
            payload.setdefault("options", {})["x_vector_only_mode"] = False
