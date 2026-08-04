from tts_audiobook_tool.constants import VOICE_ADVANCED_SUPERLABEL
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.zonos2_server_base_model import Zonos2ServerBaseModel


class VoiceZonos2ServerMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(
                state, TtsModelType.ZONOS2_SERVER,
            )

            temperature_item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="zonos2_temperature",
                default_value=Zonos2ServerBaseModel.TEMPERATURE_DEFAULT,
                min_value=Zonos2ServerBaseModel.TEMPERATURE_MIN,
                max_value=Zonos2ServerBaseModel.TEMPERATURE_MAX,
            )
            temperature_item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(temperature_item)

            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="zonos2_top_k",
                    default_value=Zonos2ServerBaseModel.TOP_K_DEFAULT,
                    min_value=Zonos2ServerBaseModel.TOP_K_MIN,
                    max_value=Zonos2ServerBaseModel.TOP_K_MAX,
                )
            )

            items.append(
                VoiceMenuShared.make_repetition_penalty_item(
                    state=state,
                    attr="zonos2_repetition_penalty",
                    default_value=Zonos2ServerBaseModel.REPETITION_PENALTY_DEFAULT,
                    min_value=Zonos2ServerBaseModel.REPETITION_PENALTY_MIN,
                    max_value=Zonos2ServerBaseModel.REPETITION_PENALTY_MAX,
                )
            )

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
