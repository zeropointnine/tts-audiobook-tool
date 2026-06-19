from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared


class VoiceQwen3ServerMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            items = []

            items.append(
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelType.QWEN3TTS_SERVER)
                )
            )

            if state.project.qwen3_voice_file_name or state.project.qwen3_voice_transcript:
                items.append(
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelType.QWEN3TTS_SERVER)
                )

            temperature_item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="qwen3_temperature",
                default_value=Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT,
                min_value=Qwen3BaseModel.TEMPERATURE_MIN,
                max_value=Qwen3BaseModel.TEMPERATURE_MAX,
            )
            temperature_item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(temperature_item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="qwen3_top_p",
                    default_value=Qwen3BaseModel.TOP_P_DEFAULT,
                )
            )

            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="qwen3_top_k",
                    default_value=Qwen3BaseModel.TOP_K_DEFAULT,
                )
            )

            items.append(
                VoiceMenuShared.make_repetition_penalty_item(
                    state=state,
                    attr="qwen3_repetition_penalty",
                    default_value=Qwen3BaseModel.REPETITION_PENALTY_DEFAULT,
                )
            )

            # Rem, sgl-omni qwen3tts does NOT support seed for now

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
