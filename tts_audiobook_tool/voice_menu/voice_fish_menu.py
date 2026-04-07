from tts_audiobook_tool.menu_util import MenuUtil, MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.fish_base_model import FishBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceFishMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.FISH)
                )
            )
            if state.project.fish_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.FISH) 
                )

            items.append( 
                VoiceMenuShared.make_temperature_item(
                    state=state,
                    attr="fish_temperature",
                    default_value=FishBaseModel.DEFAULT_TEMPERATURE,
                    min_value=0.01,
                    max_value=2.0
                )
            )

            items.append(
                VoiceMenuShared.make_seed_item(state, "fish_seed")
            )

            items.append(
                MenuItem(
                    make_menu_label("Torch compile", state.project.fish_compile_enabled),
                    lambda _, __: VoiceFishMenu.fish_compile_enabled_menu(state)
                )
            )

            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def fish_compile_enabled_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.project.fish_compile_enabled != value:
                state.project.fish_compile_enabled = value
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project)
            print_feedback(f"Set to:", str(state.project.fish_compile_enabled))

        MenuUtil.options_menu(
            state=state,
            heading_text="Torch compile",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.fish_compile_enabled,
            default_value=True,
            on_select=on_select
        )
