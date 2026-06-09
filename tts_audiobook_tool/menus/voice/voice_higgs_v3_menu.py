from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.higgs_v3_server_base_model import HiggsV3ServerBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice import VoiceMenuShared
from tts_audiobook_tool.util import ellipsize_path_for_menu, make_menu_label, truncate_pretty


class VoiceHiggsV3Menu:

    @staticmethod
    def make_voice_file_label(state: State) -> str:
        value = state.project.higgs_v3_voice_file_path
        return make_menu_label("Enter voice clone sample filepath", ellipsize_path_for_menu(value) if value else "none")

    @staticmethod
    def make_voice_transcript_label(state: State) -> str:
        value = state.project.higgs_v3_voice_transcript
        if value:
            label_value = truncate_pretty(value, 40)
        elif state.project.higgs_v3_voice_file_path:
            required = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"
            return f"Enter voice clone sample transcript {required}"
        else:
            label_value = "none"
        return make_menu_label("Enter voice clone sample transcript", label_value)

    @staticmethod
    def ask_voice_filepath(state: State) -> None:
        s = (
            "Enter voice clone reference audio path:\n"
            f"{COL_DIM}This must point to a file accessible from the running server environment"
        )
        
        ask.ask_string_and_save(
            state.project,
            s,
            "higgs_v3_voice_file_path",
            "Voice clone sample filepath set:",
        )

    @staticmethod
    def ask_voice_transcript(state: State) -> None:
        ask.ask_string_and_save(
            state.project,
            "Enter voice clone sample transcript:",
            "higgs_v3_voice_transcript",
            "Voice clone sample transcript set:",
        )

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            items = [
                MenuItem(
                    VoiceHiggsV3Menu.make_voice_file_label,
                    lambda _, __: VoiceHiggsV3Menu.ask_voice_filepath(state)
                ),
                MenuItem(
                    VoiceHiggsV3Menu.make_voice_transcript_label,
                    lambda _, __: VoiceHiggsV3Menu.ask_voice_transcript(state)
                )
            ]
            if state.project.higgs_v3_voice_file_path:
                items.append(
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelType.SERVER_HIGGS_V3)
                )

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

            if False:
                # TODO: Wait for upstream fix and then re-enable
                items.append(VoiceMenuShared.make_seed_item(state, "higgs_v3_seed"))

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)
