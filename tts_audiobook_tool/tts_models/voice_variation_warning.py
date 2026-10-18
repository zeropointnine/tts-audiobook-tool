"""The "generated voices may vary" note, for the voices a generation run uses.

One model-agnostic check, so every model gets the same rule: the note applies
when any voice in play for the run (`ProjectVoiceUtil.get_job_voice_indices`)
has no voice sample and the model can generate without one. It replaces the
per-model notes that only looked at voice 1.

A model may declare `[models.spec.ui].instructions_voice_variation_note` for
sample-less voices that do have instructions (eg AuK, whose voice description
sets the kind of voice but not a fixed speaker); otherwise the generic note is
used. Lightweight: called in both the main process and the model worker.
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


def get_voice_variation_warning(project: Project, voice_indices: Iterable[int]) -> str:
    """The note for these voice slots, or "" when none applies."""
    from tts_audiobook_tool.tts import Tts

    model_type = project.get_tts_model_type()
    if model_type.value.requires_voice or REGISTRY.voice_binding(model_type.id) is None:
        return ""
    support = Tts.get_model_support(project)
    # Samples are irrelevant under these settings (eg Qwen3-TTS CustomVoice/VoiceDesign)
    if not support.can_voice_clone(project):
        return ""
    voices = project.voice_references
    sample_less = [voices[index] for index in sorted(set(voice_indices))
                   if 0 <= index < len(voices) and not voices[index].has_sample]
    if not sample_less:
        return ""
    note = model_type.value.ui.get("instructions_voice_variation_note")
    if note and support.can_instructions(project) and any(
            voice.instructions.get(model_type.id, "").strip() for voice in sample_less):
        return f"Note: {note}"
    return f"Note: {GENERIC_VOICE_VARIATION_NOTE}"


def get_run_warnings(
        project: Project, instance: _HasWarningIssues, phrase_indices: Iterable[int] | None,
) -> list[str]:
    """Model warnings plus the voice-variation note for a run over `phrase_indices`
    (None: a literal prompt/chat with no line assignment, as in the voice pre-flight)."""
    voice_indices = ProjectVoiceUtil.get_job_voice_indices(project, phrase_indices)
    note = get_voice_variation_warning(project, voice_indices)
    # Callers previously accepted a falsy result, so None still means no warnings
    return ([note] if note else []) + list(instance.get_warning_issues(project) or [])
