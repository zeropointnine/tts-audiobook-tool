from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_lora_items import VoiceLoraMenuSpec, make_voice_lora_items
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.model.model_menu_shared import TargetPromptSpec
from tts_audiobook_tool.project import Project

if TYPE_CHECKING:
    from tts_audiobook_tool.tts_models.lora_files import LoraCheck

MODEL_ID = "breeze_tts_2_local"
INSTAVAR_URL = "https://github.com/instavar/breeze-tts2-finetuning"


def check_lora(_project: Project, lora: VoiceLora) -> LoraCheck:
    from tts_audiobook_tool.tts_models.breeze_lora_check import check_adapter
    return check_adapter(lora.target, lora.strength)


def make_lora_spec() -> VoiceLoraMenuSpec:
    from tts_audiobook_tool.tts_models.breeze_lora_source import LOCAL_FILE_MARKERS
    return VoiceLoraMenuSpec(model_id=MODEL_ID, check=check_lora, target_prompt=TargetPromptSpec(
        subject="Breeze TTS 2 LoRA (Instavar format)",
        hf_hint="Only an exported adapter (with adapter_config.json) can be loaded from Hugging Face.",
        local_hint=("An exported adapter directory (with adapter_config.json),\n"
                    "or a training checkpoint directory, eg lora/checkpoint-step-000500."),
        note=f"Train one with {INSTAVAR_URL}.",
        local_file_markers=LOCAL_FILE_MARKERS,
    ))


class VoiceBreezeMenu:
    """The shared voice-sample items, with an optional Instavar-format LoRA (see `breeze_lora`)
    and its strength in each voice's submenu (`voice_lora_items`)."""

    @staticmethod
    def menu(state: State) -> None:
        spec = make_lora_spec()

        def voice_extra_items(s: State, index: int) -> list[MenuItem]:
            return make_voice_lora_items(s, spec, index)

        def make_items(_: State) -> list[MenuItem]:
            return VoiceMenuShared.make_voice_sample_items(
                state, TtsModelType.require_by_id(MODEL_ID), voice_extra_items=voice_extra_items,
            )

        VoiceMenuShared.menu_wrapper(state, make_items)
