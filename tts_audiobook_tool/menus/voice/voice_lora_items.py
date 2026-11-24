"""Per-voice LoRA items for a voice's submenu (`Voice.loras`), for models whose
`voice_lora_mode` isn't "none". Model-agnostic: a model's voice menu supplies a
`VoiceLoraMenuSpec` (its target prompt, including which local files mark an adapter directory).

Selecting a LoRA checks it before saving: with the spec's model-free `check` when it has one
(eg `breeze_lora_check`; in this process, without loading the model), else against the loaded
model (`ModelWorker.inspect_tts_blocking` with `probe_lora`, loading it if needed). A check's
warnings are shown after saving. Strength and clear save directly. None of these reload the
model: the LoRA is applied per call, and every run checks its LoRAs against the loaded model.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import (
    LORA_STRENGTH_DEFAULT, LORA_STRENGTH_MAX, LORA_STRENGTH_MIN, VoiceLora,
)
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import (
    COL_ACCENT, COL_DEFAULT, COL_DIM_ITALICS, ellipsize_path_for_menu, make_menu_label, print_feedback, printt,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.menus.model.model_menu_shared import TargetPromptSpec
    from tts_audiobook_tool.tts_models.lora_files import LoraCheck

LABEL_LORA = "LoRA"
LABEL_LORA_STRENGTH = "LoRA strength"


@dataclass(frozen=True)
class VoiceLoraMenuSpec:
    model_id: str
    # How the adapter is asked for (`ModelMenuShared.ask_target`)
    target_prompt: TargetPromptSpec
    # Checks an adapter without the model (must be torch-free); None: probe the loaded model
    check: Callable[[Project, VoiceLora], LoraCheck] | None = None


def format_lora_strength(strength: float) -> str:
    return f"{strength:.2f}"


def make_voice_lora_items(state: State, spec: VoiceLoraMenuSpec, index: int) -> list[MenuItem]:
    """The LoRA picker (including clear), then its strength when a LoRA is set."""
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
    return items


def ask_voice_lora_target(state: State, spec: VoiceLoraMenuSpec, index: int) -> None:
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    current = state.project.voice_references[index].lora_for(spec.model_id)
    ModelMenuShared.ask_target(
        project=state.project,
        spec=spec.target_prompt,
        current_target=current.target if current else None,
        callback=lambda _, target: validate_and_set_voice_lora(state, spec, index, target),
        on_clear=lambda: clear_voice_lora(state, spec, index),
    )


def validate_and_set_voice_lora(state: State, spec: VoiceLoraMenuSpec, index: int, target: ModelTarget) -> None:
    """Checks the adapter (see the module docstring), then saves it on the voice, keeping the
    voice's current strength, and shows any warnings. Nothing is saved when it's rejected."""
    current = state.project.voice_references[index].lora_for(spec.model_id)
    lora = VoiceLora(target=target, strength=current.strength if current else LORA_STRENGTH_DEFAULT)

    if spec.check is not None:
        result = spec.check(state.project, lora)
        reason, warnings = result.error, result.warnings
    else:
        printt(f"{COL_DIM_ITALICS}Checking LoRA...")
        printt()
        reason, error = _probe_loaded_model(state, lora)
        if error:
            ask.ask_error(error)
            return
        warnings = ()
    if reason:
        ask.ask_error(f"Couldn't load LoRA: {reason}")
        return
    error = ProjectVoiceUtil.set_voice_lora_at_index_and_save(state.project, index, spec.model_id, lora)
    if error:
        ask.ask_error(error)
        return
    if warnings:
        for warning in warnings:
            printt(f"{COL_ACCENT}Warning: {COL_DEFAULT}{warning}")
        printt()
        ask.ask_enter_to_continue()
    print_feedback("LoRA set", long_pause=True)


def _probe_loaded_model(state: State, lora: VoiceLora) -> tuple[str, str]:
    """(the model's rejection reason or "", an error that prevented the check or "")."""
    from tts_audiobook_tool.model_worker import ModelWorker

    inspection, error = ModelWorker.inspect_tts_blocking(state, probe_lora=lora)
    if error or inspection is None:
        return "", f"{error}"
    reason = (inspection.metadata or {}).get("lora_probe_error")
    if reason is None:
        return "", "Couldn't check LoRA (model didn't report a result)"
    return str(reason), ""


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
