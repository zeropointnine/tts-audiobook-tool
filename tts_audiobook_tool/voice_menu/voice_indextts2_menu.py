from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import IndexTts2Protocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared

class VoiceIndexTts2Menu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """
        project = state.project

        def voice_label(_) -> str:
            if project.indextts2_voice_file_name:
                current = make_currently_string(project.get_voice_label())
            else:
                current = f"{COL_DIM}({COL_ERROR}required{COL_DIM})" # nb
            return f"Select voice clone sample {current}"

        def on_voice(_: State, __: MenuItem) -> None:
            Hint.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.INDEXTTS2)

        def make_temperature_label(_) -> str:
            value = make_parameter_value_string(
                project.indextts2_temperature, IndexTts2Protocol.DEFAULT_TEMPERATURE, 2
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                "Enter temperature (0.01 <= value <= 2.0):",
                0.01, 2.0,
                "indextts2_temperature",
                "Temperature set to:"
            )

        def make_emo_voice_label(_) -> str:
            if project.indextts2_emo_voice_file_name:
                current = make_currently_string(project.get_voice_label(is_secondary=True))
            else:
                current = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Select emotion voice sample {current}"

        def on_emo_voice(_: State, __: MenuItem) -> None:
            # TODO: disallow emo voice file == voice file (bc is default behavior anyway)
            Hint.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceMenuShared.ask_and_set_voice_file(
                state=state,
                tts_type=TtsModelInfos.INDEXTTS2,
                is_secondary=True,
                message_override="Enter emotion reference audio clip file path:"
            )

        def on_clear_emo(_: State, __: MenuItem) -> None:
            project.clear_voice_and_save(TtsModelInfos.INDEXTTS2, is_secondary=True)
            print_feedback("Cleared")

        def make_vector_label(_) -> str:
            if project.indextts2_emo_vector:
                current = make_currently_string(project.emo_vector_to_string())
            else:
                current = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Emotion vector {current}"

        def make_emo_alpha_label(_) -> str:
            value = make_parameter_value_string(
                project.indextts2_emo_alpha,
                IndexTts2Protocol.DEFAULT_EMO_VOICE_ALPHA, 2
            )
            return f"Emotion alpha (strength) {make_currently_string(value)}"

        def on_emo_alpha(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                project,
                "Enter emotion alpha (0 to 1.0):",
                0.1, 1.0,
                "indextts2_emo_alpha",
                "Emotion alpha set to:"
            )

        def make_fp16_item(_) -> str:
            value = make_parameter_value_string(project.indextts2_use_fp16, IndexTts2Protocol.DEFAULT_USE_FP16)
            return f"FP16 (smaller memory footprint) {make_currently_string(value)}"

        # Menu
        def make_items(_: State) -> list[MenuItem]:
            items = [
                MenuItem(voice_label, on_voice)
            ]
            if state.project.indextts2_voice_file_name:
                items.append( VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.INDEXTTS2) )
            items.extend([
                MenuItem(make_temperature_label, on_temperature),
                MenuItem(make_emo_voice_label, on_emo_voice)
            ])
            if state.project.indextts2_emo_voice_file_name:
                items.append( MenuItem("Clear emotion voice sample", on_clear_emo) )
            items.extend([
                MenuItem(make_vector_label, lambda _, __: ask_vector(project)),
                MenuItem(make_emo_alpha_label, on_emo_alpha),
                MenuItem(make_fp16_item, lambda _, __: VoiceIndexTts2Menu.fp16_menu(state))
            ])
            return items

        VoiceMenuShared.show_voice_menu(state, make_items)

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
    s += f'{COL_DIM}Enter \"none\" to remove{COL_DEFAULT}'
    printt(s)
    printt()
    inp = AskUtil.ask("")
    if not inp:
        return
    if inp == "none":
        value = []
    else:
        value = Project.parse_emo_vector_string(inp)
        if isinstance(value, str):
            AskUtil.ask_error(value)
            return
    project.indextts2_emo_vector = value
    project.save()
    print_feedback("Emotion vector set:", project.emo_vector_to_string())
