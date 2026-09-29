from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.menus.menu_util import MenuUtil, MenuItem
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.fish_s1_base_model import FishS1BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceFishS1Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.extend(
                VoiceMenuShared.make_voice_sample_items(state, TtsModelType.FISH_S1)
            )

            items.append(
                MenuItem(
                    make_menu_label("Torch compile", state.project.get_model_setting('fish_s1', 'compile_enabled')),
                    lambda _, __: VoiceFishS1Menu.compile_menu(state), 
                    superlabel=VOICE_ADVANCED_SUPERLABEL
                )
            )

            temperature_item = VoiceMenuShared.make_temperature_item(
                state=state,
                target=SettingRef("fish_s1", "temperature"),
                default_value=FishS1BaseModel.DEFAULT_TEMPERATURE,
                min_value=0.01,
                max_value=2.0
            )
            items.append(temperature_item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    target=SettingRef("fish_s1", "top_p"),
                    default_value=FishS1BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                VoiceMenuShared.make_repetition_penalty_item(
                    state=state,
                    target=SettingRef("fish_s1", "repetition_penalty"),
                    default_value=FishS1BaseModel.DEFAULT_REPETITION_PENALTY
                )
            )

            items.append(
                VoiceMenuShared.make_seed_item(state, SettingRef("fish_s1", "seed"))
            )

            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def compile_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.project.get_model_setting('fish_s1', 'compile_enabled') != value:
                state.project.set_model_setting('fish_s1', 'compile_enabled', value)
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project)
            print_feedback(f"Set to:", str(state.project.get_model_setting('fish_s1', 'compile_enabled')))

        MenuUtil.options_menu(
            state=state,
            heading_text="Torch compile",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.get_model_setting('fish_s1', 'compile_enabled'),
            default_value=FishS1BaseModel.DEFAULT_COMPILE_ENABLED,
            on_select=on_select
        )
