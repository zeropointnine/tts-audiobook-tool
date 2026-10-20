"""The "generated voices may vary" notes, for the voices a generation run uses.

One model-agnostic check, so every model gets the same rule: a note applies to
each voice in play for the run (`ProjectVoiceUtil.get_job_voice_indices`) that
generates without a fixed speaker, ie without a sample or preset
(`ProjectVoiceUtil.resolve_voice_source` kind "instructions" or "builtin"),
when the current settings can generate without one and have no fixed default
voice (`has_fixed_default_voice`, eg a VibeVoice LoRA). Instruction-defined voices
(eg Qwen3-TTS VoiceDesign) are eligible even when sample cloning is inactive.

A model may declare `[models.spec.ui].instructions_voice_variation_note` for
sample-less voices that do have instructions (eg AuK, whose voice description
sets the kind of voice but not a fixed speaker); those voices get that note and
the rest the generic one. Each note ends with the voice numbers it covers, eg
" (Voices 1, 3)". Lightweight: called in both the main process and the model worker.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol

from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class _HasWarningIssues(Protocol):
    def get_warning_issues(self, project: Project) -> list[str]: ...


GENERIC_VOICE_VARIATION_NOTE = "Generated voices may vary because no voice reference has been configured."


def format_voice_numbers(voice_indices: Iterable[int]) -> str:
    """Eg " (Voice 2)" or " (Voices 1, 3)" for zero-based slots; "" for none."""
    numbers = sorted({index + 1 for index in voice_indices})
    if not numbers:
        return ""
    label = "Voice" if len(numbers) == 1 else "Voices"
    return f" ({label} {', '.join(str(number) for number in numbers)})"


def get_voice_variation_warnings(project: Project, voice_indices: Iterable[int]) -> list[str]:
    """The notes for these voice slots (at most one per note text), or []."""
    from tts_audiobook_tool.tts import Tts

    model_type = project.get_tts_model_type()
    if REGISTRY.voice_binding(model_type.id) is None:
        return []
    support = Tts.get_model_support(project)
    # VoiceDesign uses instructions without a fixed speaker, even though its
    # family requires samples in Base mode. Resolve active sources below: missing
    # required samples/presets belong to validation, not a variation warning.
    if (not (support.can_voice_clone(project) or support.expects_voice_instructions(project))
            or support.has_fixed_default_voice(project)):
        return []
    voices = project.voice_references
    instructions_note = model_type.value.ui.get("instructions_voice_variation_note") or ""
    described: list[int] = []
    unset: list[int] = []
    for index in sorted(set(voice_indices)):
        if not 0 <= index < len(voices):
            continue
        kind = ProjectVoiceUtil.resolve_voice_source(project, voices[index]).kind
        if kind == "instructions" and instructions_note:
            described.append(index)
        elif kind in ("instructions", "builtin"):
            unset.append(index)
    notes = []
    if described:
        notes.append(instructions_note + format_voice_numbers(described))
    if unset:
        notes.append(GENERIC_VOICE_VARIATION_NOTE + format_voice_numbers(unset))
    return notes


def get_run_warnings(
        project: Project, instance: _HasWarningIssues, phrase_indices: Iterable[int] | None,
) -> list[str]:
    """Model warnings plus the voice-variation notes for a run over `phrase_indices`
    (None: a literal prompt/chat with no line assignment, as in the voice pre-flight)."""
    voice_indices = ProjectVoiceUtil.get_job_voice_indices(project, phrase_indices)
    notes = get_voice_variation_warnings(project, voice_indices)
    # Callers previously accepted a falsy result, so None still means no warnings
    return notes + list(instance.get_warning_issues(project) or [])
