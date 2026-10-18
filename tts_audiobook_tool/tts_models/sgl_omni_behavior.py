"""Optional programmatic layer over the declarative SGL-Omni catalog entries.

The SGL-Omni counterpart of ``audio_cpp_behavior``. The catalog declares plain
data (numeric request parameters, app-side choices, menu order, request
policies); behavior that depends on other settings (eg CosyVoice3's Mode
deciding between transcript, instructions or neither) belongs in an
``SglOmniModelBehavior`` subclass instead of another catalog key.

Subclasses are registered by catalog model ID. Most entries need none;
``get_sgl_omni_behavior`` returns the base class, which reproduces the purely
catalog-driven behavior. Hooks are narrow questions: the shared adapter still
owns request building, voice/transcript validation and seed handling.

Behaviors are constructed in the main process for menus and readiness checks,
so subclasses must stay lightweight (no heavy imports).
"""
from __future__ import annotations

from typing import ClassVar

from tts_audiobook_tool.tts_models.sgl_omni_definition import SglOmniModelDefinition

ParameterValues = dict[str, int | float | str]


class SglOmniModelBehavior:
    """Default behavior: entirely catalog-driven."""

    # Catalog parameters a subclass reads from `values`, mapped to the kind it
    # expects: "number" (a request parameter) or "choice" (an app-side choice).
    # Checked at construction so a catalog rename fails immediately rather
    # than as a KeyError mid-generation.
    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {}

    def __init__(self, definition: SglOmniModelDefinition):
        self.definition = definition
        self._check_required_parameters()

    def _check_required_parameters(self) -> None:
        where = f"{type(self).__name__} ({self.definition.spec.id})"
        for name, kind in self.REQUIRED_PARAMETERS.items():
            actual = ("number" if name in self.definition.parameters
                      else "choice" if name in self.definition.choices else None)
            if actual is None:
                raise ValueError(f"{where} requires catalog parameter {name!r}")
            if actual != kind:
                raise ValueError(f"{where} expects {name!r} to be a {kind} parameter, not {actual}")

    def uses_reference_transcript(self, values: ParameterValues) -> bool:
        """Whether a request with these resolved values reads (and so sends)
        the reference transcript. When False it is neither required nor sent.

        Go through ``ConfiguredModelSupport.uses_reference_transcript(project)``
        (or ``Tts.requires_reference_transcript``) rather than calling this
        directly. The catalog's transcript policy still governs storage and
        editing."""
        return self.definition.transcript_policy != "omitted"

    def uses_voice_instructions(self, values: ParameterValues) -> bool:
        """Whether this request consumes the selected voice's instructions."""
        return self.definition.voice_instructions

    def expects_voice_instructions(self, values: ParameterValues) -> bool:
        """Whether empty instructions deserve a (non-blocking) voice-list hint."""
        return False

    def uses_instructions_with_sample(self) -> bool:
        """Whether a voice's instructions are still sent when it has a voice
        sample. False means instructions apply only to sample-less voices."""
        return True


def get_sgl_omni_behavior(definition: SglOmniModelDefinition) -> SglOmniModelBehavior:
    """Return the behavior object for a catalog entry, keyed by catalog model ID."""
    # Imported here so subclasses can import this module's base class.
    from tts_audiobook_tool.tts_models.sgl_omni_behavior_auk import AukSglOmniBehavior
    from tts_audiobook_tool.tts_models.sgl_omni_behavior_cosyvoice3 import CosyVoice3SglOmniBehavior

    registry: dict[str, type[SglOmniModelBehavior]] = {
        "auk_sglomni": AukSglOmniBehavior,
        "auk_flash_sglomni": AukSglOmniBehavior,
        "cosyvoice3_sglomni": CosyVoice3SglOmniBehavior,
    }
    return registry.get(definition.spec.id, SglOmniModelBehavior)(definition)
