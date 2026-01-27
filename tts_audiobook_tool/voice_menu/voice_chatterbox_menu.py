from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import ChatterboxProtocol, ChatterboxType
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceChatterboxMenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        
        def make_items(_: State) -> list[MenuItem]:

            items = []
            items.append(
                MenuItem(
                    VoiceMenuShared.make_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.CHATTERBOX)
            ))
            if state.project.chatterbox_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.CHATTERBOX) 
                )
            items.append( 
                MenuItem(make_type_label, lambda _, __: ask_type(state)) 
            )
            items.append( 
                VoiceMenuShared.make_temperature_item(
                    state=state, 
                    attr="chatterbox_temperature", 
                    default_value=ChatterboxProtocol.DEFAULT_TEMPERATURE, 
                    min_value=0.01, max_value=2.0
                )
            )
            items.append( 
                MenuUtil.make_number_item(
                    state=state, 
                    attr="chatterbox_exaggeration",
                    base_label="Exaggeration",
                    default_value=ChatterboxProtocol.DEFAULT_EXAGGERATION,
                    is_minus_one_default=True,
                    num_decimals=2,
                    prompt=f"Enter value for exaggeration {COL_DIM}({0.25} to {2.0}){COL_DEFAULT}:",
                    min_value=0.25, 
                    max_value=2.0
                )
            )
            items.append( 
                MenuUtil.make_number_item(
                    state=state, 
                    attr="chatterbox_cfg",
                    base_label="CFG/pace",
                    default_value=ChatterboxProtocol.DEFAULT_CFG,
                    is_minus_one_default=True,
                    num_decimals=2,
                    prompt=f"Enter value for CFG {COL_DIM}({0.2} to {1.0}){COL_DEFAULT}:",
                    min_value=0.2, 
                    max_value=1.0
                )
            )
            items.append(
                VoiceMenuShared.make_seed_item(state, "chatterbox_seed")
            )
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)

def make_type_label(state: State) -> str:
    return make_menu_label("Chatterbox model", state.project.chatterbox_type.label)

def ask_type(state: State) -> None:

    def on_select(value: ChatterboxType) -> None:
        state.project.chatterbox_type = value
        state.project.save()
        # Sync static value
        Tts.set_model_params_using_project(state.project) 
        print_feedback("Model set to:", state.project.chatterbox_type.label)

    MenuUtil.options_menu(
        state=state,
        heading_text="Chatterbox model",
        labels=[item.label for item in list(ChatterboxType)],
        values=[item for item in list(ChatterboxType)],
        current_value=state.project.chatterbox_type,
        default_value=list(ChatterboxType)[0],
        on_select=on_select
    )
