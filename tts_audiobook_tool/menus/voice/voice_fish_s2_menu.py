from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceFishS2Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelType.FISH_S2)
                )
            )
            if state.project.fish_s2_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelType.FISH_S2) 
                )

            items.append(
                MenuItem(
                    make_menu_label("Torch compile", state.project.fish_s2_compile_enabled),
                    lambda _, __: VoiceFishS2Menu.compile_menu(state)
                )
            )

            item = MenuItem(
                VoiceMenuShared.make_rolling_continuation_label(state.project.fish_s2_rolling_cont),
                lambda _, __: VoiceMenuShared.ask_rolling_continuation(
                    state=state, 
                    attribute_name="fish_s2_rolling_cont", 
                    max_value=FishS2BaseModel.ROLLING_CONTINUATION_MAX_LENGTH, 
                    qualifier_line="Qwen3-TTS model must be of type \"base\", and batch size must be 1."
                ),
                superlabel=VOICE_ADVANCED_SUPERLABEL
            )
            items.append(item)

            temperature_item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="fish_s2_temperature",
                default_value=FishS2BaseModel.DEFAULT_TEMPERATURE,
                min_value=FishS2BaseModel.MIN_TEMPERATURE,
                max_value=FishS2BaseModel.MAX_TEMPERATURE
            )
            items.append(temperature_item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="fish_s2_top_p",
                    default_value=FishS2BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="fish_s2_top_k",
                    default_value=FishS2BaseModel.DEFAULT_TOP_K
                )
            )

            items.append(
                VoiceMenuShared.make_seed_item(state, "fish_s2_seed")
            )

            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def compile_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.project.fish_s2_compile_enabled != value:
                state.project.fish_s2_compile_enabled = value
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project)
            print_feedback(f"Set to:", str(state.project.fish_s2_compile_enabled))

        MenuUtil.options_menu(
            state=state,
            heading_text="Torch compile",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.fish_s2_compile_enabled,
            default_value=True,
            on_select=on_select
        )
