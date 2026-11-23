"""
Load-time assignment of target kinds (`ModelTarget`) to targets saved without one.

Targets saved before `target_type` existed (or by an older build, which drops it)
read with an inferred kind (`ModelTarget.from_stored`). `assign_inferred_targets`
makes those kinds explicit on the project, so the next save records them, and
reports what it assigned for a one-time notice (`make_assignment_warning`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.util import COL_ACCENT, COL_DEFAULT

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


# Models whose `parameters.target` / `parameters.target_type` hold a custom base model
MODEL_TARGET_MODEL_IDS = (
    "dots_local",
    "moss_local",
    "omnivoice_local",
    "qwen3tts_local",
    "vibevoice_local",
)


@dataclass(frozen=True)
class TargetAssignment:
    """A target whose kind was assigned; `label` names where it lives, for messages."""
    label: str
    target: ModelTarget


def _model_name(model_id: str) -> str:
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
    model_type = TtsModelType.get_by_id(model_id)
    return model_type.value.ui.get("short_name") or model_id


def assign_inferred_targets(project: Project) -> list[TargetAssignment]:
    """
    Makes every inferred target kind explicit: base model targets are rewritten as a
    typed pair, and voice LoRA targets lose their `inferred` mark (they always save
    their kind). Returns what was assigned, in a stable order; empty when nothing was
    inferred, so a second call reports nothing.
    """
    result: list[TargetAssignment] = []
    for model_id in MODEL_TARGET_MODEL_IDS:
        target = project.get_model_target(model_id)
        if target is not None and target.inferred:
            project.set_model_target(model_id, replace(target, inferred=False))
            result.append(TargetAssignment(f"{_model_name(model_id)} model", target))

    voices = list(project.voice_references)
    changed = False
    for index, voice in enumerate(voices):
        if not any(lora.target.inferred for lora in voice.loras.values()):
            continue
        loras = dict(voice.loras)
        for model_id, lora in voice.loras.items():
            if lora.target.inferred:
                loras[model_id] = replace(lora, target=replace(lora.target, inferred=False))
                result.append(TargetAssignment(f"Voice {index + 1} {_model_name(model_id)} LoRA", lora.target))
        voices[index] = replace(voice, loras=loras)
        changed = True
    if changed:
        project.voice_references = voices
    return result


def make_assignment_warning(assignments: list[TargetAssignment]) -> str:
    """The one-time notice for `assign_inferred_targets`, in the project-load warning style."""
    s = f"{COL_ACCENT}Warning/info: {COL_DEFAULT}Model targets now record whether they are a local directory\n"
    s += "or a Hugging Face repo. These were assigned automatically:\n"
    for assignment in assignments:
        target = assignment.target
        s += f"- {COL_ACCENT}{assignment.label}{COL_DEFAULT}: {target.value} ({target.kind.label})\n"
    s += "If one is wrong, re-enter it in the model settings (or, for a voice's LoRA, the voice's settings)."
    return s
