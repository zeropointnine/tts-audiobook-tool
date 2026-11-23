"""Per-voice LoRA items for a voice's submenu (`Voice.loras`), for models whose
`voice_lora_mode` isn't "none". Model-agnostic: a model's voice menu supplies a
`VoiceLoraMenuSpec` (its prompt and which local files mark an adapter directory).

Selecting a LoRA validates it against the loaded model (`ModelWorker.inspect_tts_blocking`
with `probe_lora`) before saving; strength and clear save directly. None of these
reload the model: the LoRA is applied per call.
"""
from __future__ import annotations

from dataclasses import dataclass

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import (
    LORA_STRENGTH_DEFAULT, LORA_STRENGTH_MAX, LORA_STRENGTH_MIN, VoiceLora,
)
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import (
    COL_DIM_ITALICS, ellipsize_path_for_menu, make_menu_label, print_feedback, printt,
)

LABEL_LORA = "LoRA"
LABEL_LORA_STRENGTH = "LoRA strength"
LABEL_CLEAR_LORA = "Clear LoRA"


@dataclass(frozen=True)
class VoiceLoraMenuSpec:
    model_id: str
    # Shown when asking for the target (local directory or hf repo id)
    prompt: str
    # Files that make a local directory acceptable (`target_util.exist_test`); None: the generic markers
    local_file_markers: list[str] | None = None


def format_lora_strength(strength: float) -> str:
    return f"{strength:.2f}"


def make_voice_lora_items(state: State, spec: VoiceLoraMenuSpec, index: int) -> list[MenuItem]:
    """The LoRA item, then (only with a LoRA set) its strength and Clear."""
    voices = state.project.voice_references
    if not 0 <= index < len(voices):
        return []
    lora = voices[index].lora_for(spec.model_id)
    value = ellipsize_path_for_menu(lora.target.value) if lora else "none"
    items = [MenuItem(make_menu_label(LABEL_LORA, value), lambda s, _: ask_voice_lora_target(s, spec, index))]
    if lora:
        items.append(MenuItem(
            make_menu_label(LABEL_LORA_STRENGTH, format_lora_strength(lora.strength),
                            format_lora_strength(LORA_STRENGTH_DEFAULT)),
            lambda s, _: ask_voice_lora_strength(s, spec, index),
        ))
        items.append(MenuItem(LABEL_CLEAR_LORA, lambda s, _: clear_voice_lora(s, spec, index)))
    return items


def ask_voice_lora_target(state: State, spec: VoiceLoraMenuSpec, index: int) -> None:
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    current = state.project.voice_references[index].lora_for(spec.model_id)
    ModelMenuShared.ask_target(
        project=state.project,
        prompt=spec.prompt,
        current_target=current.target.value if current else "",
        callback=lambda _, target: validate_and_set_voice_lora(
            state, spec, index, ModelTarget.from_stored(target, None)),
        local_file_markers=spec.local_file_markers,
    )


def validate_and_set_voice_lora(state: State, spec: VoiceLoraMenuSpec, index: int, target: ModelTarget | None) -> None:
    """Validates the adapter against the loaded model (loading it if needed), then saves it
    on the voice, keeping the voice's current strength. Nothing is saved when it's rejected."""
    from tts_audiobook_tool.model_worker import ModelWorker

    current = state.project.voice_references[index].lora_for(spec.model_id)
    if target is None:
        return
    lora = VoiceLora(target=target, strength=current.strength if current else LORA_STRENGTH_DEFAULT)

    printt(f"{COL_DIM_ITALICS}Checking LoRA...")
    printt()
    inspection, error = ModelWorker.inspect_tts_blocking(state, probe_lora=lora)
    if error or inspection is None:
        ask.ask_error(f"{error}")
        return
    reason = (inspection.metadata or {}).get("lora_probe_error")
    if reason is None:
        ask.ask_error("Couldn't check LoRA (model didn't report a result)")
        return
    if reason:
        ask.ask_error(f"Couldn't load LoRA: {reason}")
        return
    error = ProjectVoiceUtil.set_voice_lora_at_index_and_save(state.project, index, spec.model_id, lora)
    if error:
        ask.ask_error(error)
        return
    print_feedback("LoRA set:", target)


def ask_voice_lora_strength(state: State, spec: VoiceLoraMenuSpec, index: int) -> None:
    current = state.project.voice_references[index].lora_for(spec.model_id)
    if current is None:
        return
    printt(f"Enter LoRA strength (multiplier on the LoRA's effect), "
           f"from {LORA_STRENGTH_MIN:g} to {LORA_STRENGTH_MAX:g}:")
    value = ask.ask_input(prefill=format_lora_strength(current.strength))
    if not value:
        return
    try:
        strength = float(value)
    except (ValueError, OverflowError):
        ask.ask_error("Bad value")
        return
    if not LORA_STRENGTH_MIN <= strength <= LORA_STRENGTH_MAX:
        ask.ask_error("Out of range")
        return
    if strength == current.strength:
        return
    error = ProjectVoiceUtil.set_voice_lora_at_index_and_save(
        state.project, index, spec.model_id, VoiceLora(target=current.target, strength=strength))
    if error:
        ask.ask_error(error)
    else:
        print_feedback(f"{LABEL_LORA_STRENGTH} set to: {format_lora_strength(strength)}")


def clear_voice_lora(state: State, spec: VoiceLoraMenuSpec, index: int) -> None:
    error = ProjectVoiceUtil.set_voice_lora_at_index_and_save(state.project, index, spec.model_id, None)
    if error:
        ask.ask_error(error)
    else:
        print_feedback("Cleared LoRA")
