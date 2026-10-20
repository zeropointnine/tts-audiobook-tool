"""Optional programmatic layer over the declarative audio.cpp catalog entries.

The catalog declares plain data: parameters, bounds, request placement, menu
order and simple shared flags. Behavior that depends on other settings (eg
CosyVoice3's Mode deciding whether the transcript is read) belongs in an
``AudioCppModelBehavior`` subclass instead of another catalog key.

Subclasses are registered by catalog model ID (not audio.cpp's server-side
`family`). Most entries need no subclass; ``get_audio_cpp_behavior`` returns
the base class, which reproduces the purely catalog-driven behavior. Hooks are
narrow questions or finishing edits: the shared adapter still owns request
building, voice/transcript validation and seed handling.

Behaviors are constructed in the main process for menus and readiness checks,
so subclasses must stay lightweight (no heavy imports).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.audio_cpp_definition import (
    AudioCppModelDefinition,
    AudioCppParameter,
    AudioCppTextParameter,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project

ParameterValues = dict[str, int | float | str]


class AudioCppModelBehavior:
    """Default behavior: entirely catalog-driven."""

    # Catalog parameters a subclass reads from `values`, mapped to the kind it
    # expects: "number", "text" (free-form string) or "choice" (string with
    # declared choices). Checked at construction so a catalog rename fails
    # immediately rather than as a KeyError mid-generation.
    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {}

    def __init__(self, definition: AudioCppModelDefinition):
        self.definition = definition
        self._check_required_parameters()

    def _check_required_parameters(self) -> None:
        where = f"{type(self).__name__} ({self.definition.spec.id})"
        for name, kind in self.REQUIRED_PARAMETERS.items():
            parameter = self.definition.parameters.get(name)
            if parameter is None:
                raise ValueError(f"{where} requires catalog parameter {name!r}")
            actual = (
                "number" if isinstance(parameter, AudioCppParameter)
                else "choice" if isinstance(parameter, AudioCppTextParameter) and parameter.choices
                else "text"
            )
            if actual != kind:
                raise ValueError(f"{where} expects {name!r} to be a {kind} parameter, not {actual}")

    def uses_reference_transcript(self, values: ParameterValues) -> bool:
        """Whether a request with these resolved parameter values reads the
        reference transcript. When False, the transcript is neither required
        nor sent.

        Code outside this layer should not call this directly: it goes
        through ``AudioCppModelSupport.uses_reference_transcript(project)``
        (or ``Tts.requires_reference_transcript``), the single runtime
        authority shared by the adapter, voice pre-flight and server startup.
        The catalog's static ``reference_transcript`` flag only governs
        transcript storage and editing."""
        return self.definition.reference_transcript

    def uses_voice_instructions(self, values: ParameterValues) -> bool:
        """Whether this request consumes the selected voice's instructions."""
        return self.definition.voice_instructions

    def expects_voice_instructions(self, values: ParameterValues) -> bool:
        """Whether empty instructions deserve a (non-blocking) voice-list hint."""
        return False

    def can_voice_clone(self, values: ParameterValues) -> bool:
        """Whether the selected configuration consumes reference audio."""
        return self.definition.spec.can_voice_clone

    def can_voice_presets(self, values: ParameterValues) -> bool:
        """Whether a per-voice preset selects a built-in speaker."""
        return False

    def requires_voice_preset(self, values: ParameterValues) -> bool:
        return False

    def resolve_language(self, language: str) -> str:
        """Translate the catalog-normalized project language into a model hint."""
        return language

    def get_ignored_parameters(self, values: ParameterValues) -> frozenset[str]:
        """Catalog parameters the server does not read under these values;
        they are left out of the request (their stored values are kept)."""
        return frozenset()

    def adjust_payload(self, payload: dict, values: ParameterValues) -> None:
        """Escape hatch: last edit to a fully built request payload, in place.

        Prefer a narrower hook (eg ``get_ignored_parameters``). An override
        must preserve adapter-owned fields (``model``, ``input``,
        ``response_format``, ``seed``, ``voice_ref``, ``reference_text``):
        the adapter owns model identity, the resolved seed, the voice
        reference and the transcript decision, and raises if any of them
        changes here."""

    def get_blocking_issues(self, project: Project, values: ParameterValues) -> list[ReadinessIssue]:
        """Model-specific issues that prevent generation, given valid resolved
        values. Runs on menu redraws, so it must stay cheap (no I/O)."""
        return []

    def get_warning_issues(self, project: Project, values: ParameterValues) -> list[str]:
        """Model-specific pre-generation warnings, given valid resolved values."""
        return []


def get_audio_cpp_behavior(definition: AudioCppModelDefinition) -> AudioCppModelBehavior:
    """Return the behavior object for a catalog entry, keyed by catalog model ID."""
    # Imported here so subclasses can import this module's base class.
    from tts_audiobook_tool.tts_models.audio_cpp_behavior_cosyvoice3 import CosyVoice3Behavior
    from tts_audiobook_tool.tts_models.audio_cpp_behavior_fireredtts3 import FireRedTts3Behavior

    from tts_audiobook_tool.tts_models.audio_cpp_behavior_qwen3 import Qwen3Behavior

    registry: dict[str, type[AudioCppModelBehavior]] = {
        "cosyvoice3_audiocpp": CosyVoice3Behavior,
        "fireredtts3_audiocpp": FireRedTts3Behavior,
        "qwen3tts_audiocpp": Qwen3Behavior,
    }
    return registry.get(definition.spec.id, AudioCppModelBehavior)(definition)
