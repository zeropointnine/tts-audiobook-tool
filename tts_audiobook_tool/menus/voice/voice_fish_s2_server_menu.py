from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.fish_s2_server_base_model import FishS2ServerBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceFishS2ServerMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []

            # server voice path and voice transcript
            path_item, transcript_item = VoiceMenuShared.make_manual_voice_menu_items(
                state, "fish_s2_server_voice_target", "fish_s2_server_voice_transcript", is_required=False
            )
            items.append(path_item)
            items.append(transcript_item)

            # clear voice
            if state.project.higgs_v3_voice_target or state.project.higgs_v3_voice_transcript:
                items.append(
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelType.FISH_S2_SERVER)
                )

            # Hyperparams
            # Shares project attributes as local inference version, TtsModelType.FISH_S2

            temperature_item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="fish_s2_temperature",
                default_value=FishS2BaseModel.TEMPERATURE_DEFAULT,
                min_value=FishS2BaseModel.TEMPERATURE_MIN,
                max_value=FishS2BaseModel.TEMPERATURE_MAX,
            )
            temperature_item.superlabel = "Advanced"
            items.append(temperature_item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="fish_s2_top_p",
                    default_value=FishS2BaseModel.TOP_P_DEFAULT
                )
            )

            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="fish_s2_top_k",
                    default_value=FishS2BaseModel.TOP_K_DEFAULT,
                    max_value=FishS2ServerBaseModel.TOP_K_MAX
                )
            )

            # Rem, sgl-omni fish s2 does NOT support seed

            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)
