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
    AudioCppMenuControl,
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

    # Menu-facing questions. These answer *what* to show; the menu layer owns
    # how it is drawn, so this module never imports menu code. Like the
    # readiness hooks they run on every redraw, so they must stay cheap.

    def is_control_visible(self, control: AudioCppMenuControl, values: ParameterValues) -> bool:
        """Whether a settings control appears in its menu, given valid resolved
        values. Hiding a control does not change its stored value."""
        return True

    def get_required_label(self, name: str, values: ParameterValues) -> str | None:
        """Status text (eg "required for Instruct") shown in place of
        "(optional)" when an optional-looking control such as free-text
        instructions is currently required and empty; None when it is
        optional. Should agree with ``get_blocking_issues``, which is what
        actually stops generation."""
        return None


def get_audio_cpp_behavior(definition: AudioCppModelDefinition) -> AudioCppModelBehavior:
    """Return the behavior object for a catalog entry, keyed by catalog model ID."""
    # Imported here so subclasses can import this module's base class.
    from tts_audiobook_tool.tts_models.audio_cpp_behavior_cosyvoice3 import CosyVoice3Behavior

    registry: dict[str, type[AudioCppModelBehavior]] = {
        "cosyvoice3_audiocpp": CosyVoice3Behavior,
    }
    return registry.get(definition.spec.id, AudioCppModelBehavior)(definition)
