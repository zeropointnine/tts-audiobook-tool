from tts_audiobook_tool.menu_util import MenuUtil, MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceFishS1Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.FISH_S1)
                )
            )
            if state.project.fish_s1_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.FISH_S1) 
                )

            items.append(
                MenuItem(
                    make_menu_label("Torch compile", state.project.fish_s1_compile_enabled),
                    lambda _, __: VoiceFishS1Menu.compile_menu(state)
                )
            )

            items.append(
                VoiceMenuShared.make_temperature_item(
                    state=state,
                    attr="fish_s1_temperature",
                    default_value=FishS1BaseModel.DEFAULT_TEMPERATURE,
                    min_value=0.01,
                    max_value=2.0
                )
            )

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="fish_s1_top_p",
                    default_value=FishS1BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                VoiceMenuShared.make_repetition_penalty_item(
                    state=state,
                    attr="fish_s1_repetition_penalty",
                    default_value=FishS1BaseModel.DEFAULT_REPETITION_PENALTY
                )
            )

            items.append(
                VoiceMenuShared.make_seed_item(state, "fish_s1_seed")
            )

            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def compile_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.project.fish_s1_compile_enabled != value:
                state.project.fish_s1_compile_enabled = value
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project)
            print_feedback(f"Set to:", str(state.project.fish_s1_compile_enabled))

        MenuUtil.options_menu(
            state=state,
            heading_text="Torch compile",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.fish_s1_compile_enabled,
            default_value=FishS1BaseModel.DEFAULT_COMPILE_ENABLED,
            on_select=on_select
        )
