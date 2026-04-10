from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
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
                    VoiceMenuShared.make_resolved_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.CHATTERBOX)
            ))
            if state.project.chatterbox_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.CHATTERBOX) 
                )

            items.append( 
                MenuItem(make_type_label, lambda _, __: ask_type(state)) 
            )

            if state.project.chatterbox_type == ChatterboxType.MULTILINGUAL:
                items.append( 
                    MenuUtil.make_number_item(
                        state=state, 
                        attr="chatterbox_exaggeration",
                        base_label="Exaggeration",
                        default_value=ChatterboxBaseModel.DEFAULT_EXAGGERATION,
                        is_minus_one_default=True,
                        num_decimals=2,
                        prompt=f"Enter value for exaggeration {COL_DIM}({0.25} to {2.0}){COL_DEFAULT}:",
                        min_value=0.25, 
                        max_value=2.0
                    )
                )

            if state.project.chatterbox_type == ChatterboxType.MULTILINGUAL:
                items.append( 
                    MenuUtil.make_number_item(
                        state=state, 
                        attr="chatterbox_cfg",
                        base_label="CFG/pace",
                        default_value=ChatterboxBaseModel.DEFAULT_CFG,
                        is_minus_one_default=True,
                        num_decimals=2,
                        prompt=f"Enter value for CFG {COL_DIM}({0.2} to {1.0}){COL_DEFAULT}:",
                        min_value=0.2, 
                        max_value=1.0
                    )
                )

            item = VoiceMenuShared.make_temperature_item(
                state=state, 
                attr="chatterbox_temperature", 
                default_value=ChatterboxBaseModel.DEFAULT_TEMPERATURE, 
                min_value=0.01, max_value=2.0
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="chatterbox_top_p",
                    default_value=ChatterboxBaseModel.DEFAULT_TOP_P
                )
            )

            if state.project.chatterbox_type == ChatterboxType.TURBO:
                items.append(
                    VoiceMenuShared.make_top_k_item(
                        state=state,
                        attr="chatterbox_turbo_top_k",
                        default_value=ChatterboxBaseModel.DEFAULT_TOP_K
                    )
                )

            # Repetition penalty - using separate values for each variant
            match state.project.chatterbox_type:
                case ChatterboxType.MULTILINGUAL:
                    qual = "Multilingual"
                    attr = "chatterbox_ml_repetition_penalty"
                    default_value = ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_ML
                case ChatterboxType.TURBO:
                    qual = "Turbo"
                    attr = "chatterbox_turbo_repetition_penalty"
                    default_value = ChatterboxBaseModel.DEFAULT_REPETITION_PENALTY_TURBO            
            rep_min = REPETITION_PENALTY_MIN_DEFAULT
            rep_max = REPETITION_PENALTY_MAX_DEFAULT

            item = MenuUtil.make_number_item(
                state=state,
                attr=attr,
                base_label=f"Repetition penalty ({qual})", 
                default_value=default_value,
                is_minus_one_default=True,
                num_decimals=2,
                prompt=f"Enter repetition penalty {COL_DIM}({rep_min} to {rep_max}){COL_DEFAULT}:",
                min_value=rep_min,
                max_value=rep_max
            )
            items.append(item)

            items.append(
                VoiceMenuShared.make_seed_item(state, "chatterbox_seed")
            )
            return items
        
        VoiceMenuShared.menu_wrapper(state, make_items)

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
