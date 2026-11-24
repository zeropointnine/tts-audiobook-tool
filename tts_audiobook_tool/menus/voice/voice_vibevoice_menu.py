from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_lora_items import VoiceLoraMenuSpec, make_voice_lora_items
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.model.model_menu_shared import TargetPromptSpec

if TYPE_CHECKING:
    from tts_audiobook_tool.tts_models.lora_files import LoraCheck

MODEL_ID = "vibevoice_local"


def check_lora(project: Project, lora: VoiceLora) -> LoraCheck:
    # Against the base model the project would load (only its config.json is read)
    from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
    from tts_audiobook_tool.tts_models.vibevoice_lora_check import check_adapter
    base_target = project.get_model_target(MODEL_ID) or VibeVoiceBaseModel.DEFAULT_TARGET
    return check_adapter(lora.target, base_target, lora.strength)


def make_lora_spec() -> VoiceLoraMenuSpec:
    from tts_audiobook_tool.tts_models.vibevoice_lora_check import LOCAL_FILE_MARKERS
    return VoiceLoraMenuSpec(model_id=MODEL_ID, check=check_lora, target_prompt=TargetPromptSpec(
        subject="VibeVoice LoRA",
        hf_example="vibevoice-community/klett",
        local_hint=("An adapter directory (eg output/checkpoint-900) contains adapter_config.json and "
                    "adapter_model.safetensors,\ndirectly or in its lora/ subdirectory."),
        local_file_markers=LOCAL_FILE_MARKERS,
    ))


class VoiceVibeVoiceMenu:
    """The shared voice-sample items, with an optional VibeVoice LoRA (see `vibevoice_lora`)
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
