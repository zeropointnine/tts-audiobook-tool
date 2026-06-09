from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.higgs_v3_server_base_model import HiggsV3ServerBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice import VoiceMenuShared


class VoiceHiggsV3Menu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:

            items = []

            # Server voice path and voice transcript
            path_item, transcript_item = VoiceMenuShared.make_manual_voice_menu_items(
                state, "higgs_v3_voice_target", "higgs_v3_voice_transcript", is_required=False
            )
            items.append(path_item)
            items.append(transcript_item)

            # Clear voice
            if state.project.higgs_v3_voice_target or state.project.higgs_v3_voice_transcript:
                items.append(
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelType.HIGGS_V3_SERVER)
                )

            # Supported Hyperparams
            item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="higgs_v3_temperature",
                default_value=HiggsV3ServerBaseModel.DEFAULT_TEMPERATURE,
                min_value=0.01,
                max_value=HiggsV3ServerBaseModel.MAX_TEMPERATURE,
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="higgs_v3_top_p",
                    default_value=HiggsV3ServerBaseModel.DEFAULT_TOP_P,
                )
            )
            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="higgs_v3_top_k",
                    default_value=HiggsV3ServerBaseModel.DEFAULT_TOP_K,
                )
            )

            # TODO: Wait for upstream fix and then re-enable
            # items.append(VoiceMenuShared.make_seed_item(state, "higgs_v3_seed"))

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
