"""Validated audio.cpp family definitions from the application catalog."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from tts_audiobook_tool.tts_models.model_catalog import (
    CATALOG_PATH,
    load_catalog,
    parse_audio_cpp_declaration,
    parse_spec,
)
from tts_audiobook_tool.tts_models.catalog_settings import parse_model_settings
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelSpec


@dataclass(frozen=True)
class AudioCppParameter:
    model_id: str
    name: str
    type: str
    default: int | float
    min: int | float
    max: int | float
    default_sentinel: int | float | None = None
    request_key: str = ""
    target: str = "top_level"
    input_prompt_suffix: str = ""

    def validate(self, value: object) -> int | float:
        typ = int if self.type == "int" else (int, float)
        if isinstance(value, bool) or not isinstance(value, typ) or not math.isfinite(value) or not self.min <= value <= self.max:
            raise ValueError(f"{self.model_id}.{self.name} must be a {self.type} between {self.min} and {self.max}")
        return value


@dataclass(frozen=True)
class AudioCppTextParameter:
    """A free-text request control, eg OmniVoice's voice-design `instruction`.

    audio.cpp has no text control that is not read from the request's
    ``options`` map, so ``target`` is fixed rather than declared, and there are
    no numeric bounds to validate against.
    """

    model_id: str
    name: str
    default: str
    request_key: str = ""
    type: str = "str"
    target: str = "options"

    def validate(self, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{self.model_id}.{self.name} must be a string")
        return value


AudioCppParameterTypes = AudioCppParameter | AudioCppTextParameter


@dataclass(frozen=True)
class AudioCppMenuControl:
    kind: str
    # Voice samples are implicitly routed; settings always declare a target.
    target_menu: str | None
    parameter: str = ""
    label: str = ""


@dataclass(frozen=True)
class AudioCppModelDefinition:
    spec: TtsModelSpec
    parameters: dict[str, AudioCppParameterTypes]
    menu: tuple[AudioCppMenuControl, ...]
    family: str
    tasks: tuple[str, ...]
    mode: str
    session_options: dict[str, Any]
    reference_transcript: bool = False
    # Whether a voice sample must exist before generation. A family served by a
    # route that also accepts reference-less requests (Breeze / OmniVoice voice
    # design or auto voice) declares false; a missing configured sample is still
    # blocking either way.
    voice_required: bool = True
    # Family-specific segment-size recommendation; None means the shared default.
    max_words_range_reco: tuple[int, int, str] | None = None
    # Constants pinned into every request's `options`; not project settings.
    request_options: dict[str, int | float] = field(default_factory=dict)
    settings: tuple[dict[str, Any], ...] = ()
    # MOSS prompt templates use full language names; other families keep ISO codes.
    # "omit" sends no language at all (families that auto-detect, e.g. Fish S2-Pro).
    language_policy: str = "project_code"
    # Community MOSS reads options.language, while Local reads Transcript.language.
    language_target: str = "top_level"
    music_and_trim: bool = False


@dataclass(frozen=True)
class AudioCppDefinitions:
    models: dict[str, AudioCppModelDefinition]
    fingerprint: str


def load_audio_cpp_definitions(path: Path = CATALOG_PATH) -> AudioCppDefinitions:
    """Read only audio.cpp entries; reject malformed settings before installation."""
    _, entries, fingerprint = load_catalog(path)
    models: dict[str, AudioCppModelDefinition] = {}
    for entry in entries:
        if entry.get("backend_kind") != TtsBackendKind.AUDIO_CPP.value:
            continue
        spec = parse_spec(entry)
        where = f"audio.cpp model {spec.id}"
        if spec.id in models:
            raise ValueError(f"{where}: duplicate model ID")
        if spec.can_stream:
            raise ValueError(f"{where}: audio.cpp families do not stream")
        declaration = parse_audio_cpp_declaration(entry, where)
        if spec.requires_voice != declaration["voice_required"]:
            raise ValueError(f"{where}: voice requirement disagrees with its declaration")
        parameters: dict[str, AudioCppParameterTypes] = {}
        for name, raw in declaration["parameters"].items():
            if raw["type"] == "str":
                parameters[name] = AudioCppTextParameter(spec.id, name, raw["default"], raw["request_key"])
            else:
                parameters[name] = AudioCppParameter(
                    spec.id, name, raw["type"], raw["default"], raw["min"], raw["max"],
                    raw["default_sentinel"], raw["request_key"], raw["target"], raw["input_prompt_suffix"],
                )
        menu = tuple(
            AudioCppMenuControl(item["kind"], item.get("target_menu"), item.get("parameter", ""), item.get("label", ""))
            for item in declaration["menu"]
        )
        models[spec.id] = AudioCppModelDefinition(
            spec, parameters, menu, declaration["family"], declaration["tasks"], declaration["mode"],
            declaration["session_options"], declaration["reference_transcript"],
            declaration["voice_required"],
            declaration["max_words_range_reco"], declaration["request_options"],
            parse_model_settings(entry),
            language_policy=declaration["language_policy"],
            language_target=declaration["language_target"],
            music_and_trim=declaration["music_and_trim"],
        )
    return AudioCppDefinitions(models, fingerprint)
