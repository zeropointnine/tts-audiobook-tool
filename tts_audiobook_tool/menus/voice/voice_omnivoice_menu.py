from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_lora_items import VoiceLoraMenuSpec, make_voice_lora_items
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.model.model_menu_shared import TargetPromptSpec

MODEL_ID = "omnivoice_local"


def make_lora_spec() -> VoiceLoraMenuSpec:
    from tts_audiobook_tool.tts_models.omnivoice_lora import ADAPTER_FILE_NAMES
    return VoiceLoraMenuSpec(model_id=MODEL_ID, target_prompt=TargetPromptSpec(
        subject="OmniVoice LoRA adapter",
        hf_example="My-Weird-Prompts/Herman-LoRA-V3",
        local_hint="An adapter directory (eg checkpoint-900) contains adapter_config.json and adapter_model.safetensors.",
        note="A LoRA trained on one speaker works without a voice sample.",
        local_file_markers=ADAPTER_FILE_NAMES,
    ))


class VoiceOmniVoiceMenu:
    """The shared voice-sample items, with an optional OmniVoice LoRA (see `omnivoice_lora`)
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
