from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_lora_items import VoiceLoraMenuSpec, make_voice_lora_items
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DIM

MODEL_ID = "omnivoice_local"


def make_lora_prompt() -> str:
    prompt = "Enter huggingface repo id or local directory path to OmniVoice LoRA adapter"
    prompt += f"\n{COL_DIM}Eg, \"My-Weird-Prompts/Herman-LoRA-V3\", \"/path/to/checkpoint-900\"."
    prompt += f"\n{COL_DIM}A local adapter directory contains adapter_config.json and adapter_model.safetensors."
    prompt += f"\n{COL_DIM}For a hf subdirectory, enter, for example: \"owner/repo/sub/dir\"."
    prompt += f"\n{COL_DIM}A LoRA trained on one speaker works without a voice sample."
    return prompt


def make_lora_spec() -> VoiceLoraMenuSpec:
    from tts_audiobook_tool.tts_models.omnivoice_lora import ADAPTER_FILE_NAMES
    return VoiceLoraMenuSpec(model_id=MODEL_ID, prompt=make_lora_prompt(), local_file_markers=ADAPTER_FILE_NAMES)


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
