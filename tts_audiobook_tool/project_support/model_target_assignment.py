"""
Load-time assignment of target kinds (`ModelTarget`) to targets saved without one.

Targets saved before `target_type` existed (or by an older build, which drops it)
read with an inferred kind (`ModelTarget.from_stored`). `assign_inferred_targets`
makes those kinds explicit on the project, so the next save records them. This is
silent: the inferred kind always matches how the app used to read the target.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

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

# Models whose old loader read an untyped `owner/repo` target as hf even when an
# identically named local directory existed (`infer_legacy_kind`'s `hf_first`)
LEGACY_HF_FIRST_MODEL_IDS = ("moss_local",)


def assign_inferred_targets(project: Project) -> bool:
    """
    Makes every inferred target kind explicit: base model targets are rewritten as a
    typed pair, and voice LoRA targets lose their `inferred` mark (they always save
    their kind). Returns whether anything was assigned, so the caller knows to save;
    a second call returns False.
    """
    changed = False
    for model_id in MODEL_TARGET_MODEL_IDS:
        target = project.get_model_target(model_id)
        if target is not None and target.inferred:
            project.set_model_target(model_id, replace(target, inferred=False))
            changed = True

    voices = list(project.voice_references)
    voices_changed = False
    for index, voice in enumerate(voices):
        if not any(lora.target.inferred for lora in voice.loras.values()):
            continue
        loras = {
            model_id: replace(lora, target=replace(lora.target, inferred=False))
            for model_id, lora in voice.loras.items()
        }
        voices[index] = replace(voice, loras=loras)
        voices_changed = True
    if voices_changed:
        project.voice_references = voices
    return changed or voices_changed
