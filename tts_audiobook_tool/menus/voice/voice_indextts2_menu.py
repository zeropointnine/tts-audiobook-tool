from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceIndexTts2Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        project = state.project

        def voice_label(_) -> str:
            if project.indextts2_voice_file_name:
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(project))
            else:
                currently = f"{COL_DIM}({COL_ERROR}required{COL_DIM})" # nb
            return f"Select voice clone sample {currently}"

        def on_voice(_: State, __: MenuItem) -> None:
            hints.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceMenuShared.ask_and_set_voice_file(state, TtsModelType.INDEXTTS2)

        def make_emo_voice_label(_) -> str:
            if project.indextts2_emo_voice_file_name:
                value = ellipsize_path_for_menu(project.indextts2_emo_voice_file_name)
                currently = make_currently_string(value)
            else:
                currently = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Select emotion voice sample {currently}"

        def on_emo_voice(_: State, __: MenuItem) -> None:
            # TODO: disallow emo voice file == voice file (bc is default behavior anyway)
            hints.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceMenuShared.ask_and_set_voice_file(
                state=state,
                tts_type=TtsModelType.INDEXTTS2,
                is_secondary=True,
                message_override="Enter emotion reference audio clip file path:"
            )

        def on_clear_emo(_: State, __: MenuItem) -> None:
            ProjectVoiceUtil.clear_voice_and_save(project, TtsModelType.INDEXTTS2, is_secondary=True)
            print_feedback("Cleared")

        def make_vector_label(_) -> str:
            if project.indextts2_emo_vector:
                current = make_currently_string(ProjectVoiceUtil.emo_vector_to_string(project))
            else:
                current = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Emotion vector {current}"

        def make_fp16_item(_) -> str:
            value = make_parameter_value_string(project.indextts2_use_fp16, IndexTts2BaseModel.DEFAULT_USE_FP16)
            return f"FP16 (smaller memory footprint) {make_currently_string(value)}"

        # Menu
        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(voice_label, on_voice)
            )
            if state.project.indextts2_voice_file_name:
                items.append( 
                    VoiceMenuShared.make_clear_voice_item(state, TtsModelType.INDEXTTS2) 
                )
            items.append(
                MenuItem(make_emo_voice_label, on_emo_voice)
            )
            if state.project.indextts2_emo_voice_file_name:
                items.append( 
                    MenuItem("Clear emotion voice sample", on_clear_emo) 
                )
            items.append(
                MenuItem(make_vector_label, lambda _, __: ask_vector(project))
            )
            items.append(
                MenuUtil.make_number_item(
                    state=state,
                    attr="indextts2_emo_alpha",
                    base_label="Emotion alpha (strength)",
                    default_value=IndexTts2BaseModel.DEFAULT_EMO_VOICE_ALPHA,
                    is_minus_one_default=True,
                    num_decimals=2,
                    prompt=f"Enter emotion alpha {COL_DIM}({0.01} to {1.0}){COL_DEFAULT}:",
                    min_value=0.01,
                    max_value=1.0
                )
            )
            items.append(
                MenuItem(make_fp16_item, lambda _, __: VoiceIndexTts2Menu.fp16_menu(state))
            )

            item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="indextts2_temperature",
                default_value=IndexTts2BaseModel.DEFAULT_TEMPERATURE,
                min_value=0.01,
                max_value=2.0
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            items.append(
                VoiceMenuShared.make_top_p_item(
                    state=state,
                    attr="indextts2_top_p",
                    default_value=IndexTts2BaseModel.DEFAULT_TOP_P
                )
            )

            items.append(
                VoiceMenuShared.make_top_k_item(
                    state=state,
                    attr="indextts2_top_k",
                    default_value=IndexTts2BaseModel.DEFAULT_TOP_K
                )
            )

            items.append(VoiceMenuShared.make_seed_item(state, "indextts2_seed"))

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def fp16_menu(state: State) -> None:

        def on_item(_: State, item: MenuItem) -> bool:
            if state.project.indextts2_use_fp16 != item.data:
                state.project.indextts2_use_fp16 = item.data
                state.project.save()
                # Sync static value
                Tts.set_model_params_using_project(state.project) 
            print_feedback(f"FP16 set to:", str(state.project.indextts2_use_fp16))
            return True

        items = [
            MenuItem("True", on_item, data=True),
            MenuItem("False", on_item, data=False)
        ]
        MenuUtil.menu(
            state,
            "FP16",
            items,
            one_shot=True
        )

# ---

def ask_vector(project: Project) -> None:

    printt("Enter emotion vector:")
    printt()
    s = "This should be a list of eight numbers between 0-1 corresponding to:\n"
    s += "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm\n"
    s += f'{COL_DIM}Eg: "0, 0.8, 0, 0, 0.2, 0, 0, 0" = very angry, slightly disgusted{COL_DEFAULT}\n'
    s += f'{COL_DIM}Enter \"none\" to clear{COL_DEFAULT}'
    printt(s)
    printt()
    inp = ask.ask("")
    if not inp:
        return
    if inp == "none":
        value = []
    else:
        value = ProjectUtil.parse_emo_vector_string(inp)
        if isinstance(value, str):
            ask.ask_error(value)
            return
    project.indextts2_emo_vector = value
    project.save()
    print_feedback("Emotion vector set:", ProjectVoiceUtil.emo_vector_to_string(project))
