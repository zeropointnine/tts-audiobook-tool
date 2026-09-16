from tts_audiobook_tool.constants import VOICE_ADVANCED_SUPERLABEL
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.auk_server_base_model import AuKServerBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DEFAULT, COL_DIM


class VoiceAuKServerMenu:
    """Shared voice settings menu for AuK and AuK-Flash."""

    @staticmethod
    def menu(state: State, model_type: TtsModelType) -> None:
        if model_type not in {
            TtsModelType.AUK_SERVER,
            TtsModelType.AUK_FLASH_SERVER,
        }:
            raise ValueError(f"Unsupported AuK server type: {model_type}")

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(state, model_type)

            speed_item = MenuUtil.make_number_item(
                state=state,
                attr="auk_speed",
                base_label="Speech speed",
                default_value=AuKServerBaseModel.SPEED_DEFAULT,
                is_minus_one_default=False,
                num_decimals=2,
                prompt=(
                    "Enter speech speed (model-generated, not post-processed; higher is faster) "
                    f"{COL_DIM}({AuKServerBaseModel.SPEED_MIN} to "
                    f"{AuKServerBaseModel.SPEED_MAX}){COL_DEFAULT}:"
                ),
                min_value=AuKServerBaseModel.SPEED_MIN,
                max_value=AuKServerBaseModel.SPEED_MAX,
            )
            speed_item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(speed_item)

            seed_item = VoiceMenuShared.make_seed_item(
                state, "auk_seed", add_batch_warning=True
            )
            items.append(seed_item)

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
