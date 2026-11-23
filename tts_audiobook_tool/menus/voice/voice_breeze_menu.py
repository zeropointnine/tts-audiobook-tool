from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_lora_items import VoiceLoraMenuSpec, make_voice_lora_items
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DIM

MODEL_ID = "breeze_tts_2_local"
INSTAVAR_URL = "https://github.com/instavar/breeze-tts2-finetuning"


def make_lora_prompt() -> str:
    prompt = "Enter local directory path or huggingface repo id of a Breeze TTS 2 LoRA (Instavar format)"
    prompt += f"\n{COL_DIM}Eg, an exported adapter directory (with adapter_config.json),"
    prompt += f"\n{COL_DIM}or a training checkpoint directory, eg \"/path/to/lora/checkpoint-step-000500\"."
    prompt += f"\n{COL_DIM}Train one with {INSTAVAR_URL}."
    return prompt


def make_lora_spec() -> VoiceLoraMenuSpec:
    from tts_audiobook_tool.tts_models.breeze_lora import LOCAL_FILE_MARKERS
    return VoiceLoraMenuSpec(model_id=MODEL_ID, prompt=make_lora_prompt(), local_file_markers=LOCAL_FILE_MARKERS)


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
