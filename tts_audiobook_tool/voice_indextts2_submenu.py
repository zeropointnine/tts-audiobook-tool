from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import IndexTts2Protocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class VoiceIndexTts2Submenu:

    @staticmethod
    def menu(state: State) -> None:
        """
        """

        project = state.project

        # 1
        def make_voice_label(_) -> str:
            if project.indextts2_voice_file_name:
                current = make_currently_string(project.get_voice_label())
            else:
                current = f"{COL_DIM}({COL_ERROR}required{COL_DIM})" # nb
            return f"Select voice clone sample {current}"

        def on_voice(_, __) -> None:
            AppUtil.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceSubmenuShared.ask_and_set_voice_file(state, TtsModelInfos.INDEXTTS2)

        # 2
        def on_clear_voice(_, __) -> None:
            project.clear_voice_and_save(TtsModelInfos.INDEXTTS2)
            printt_set("Cleared")

        # 3
        def make_temperature_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.indextts2_temperature, IndexTts2Protocol.DEFAULT_TEMPERATURE, 2
            )
            return f"Temperature {make_currently_string(value)}"

        def on_temperature(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter temperature (0.01 <= value <= 2.0):",
                0.01, 2.0,
                "indextts2_temperature",
                "Temperature set to:"
            )

        # 4
        def make_emo_voice_label(_) -> str:
            if project.indextts2_emo_voice_file_name:
                current = make_currently_string(project.get_voice_label(is_secondary=True))
            else:
                current = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Select emotion voice sample {current}"

        def on_emo_voice(_, __) -> None:
            # TODO: disallow emo voice file == voice file (bc is default behavior anyway)
            AppUtil.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceSubmenuShared.ask_and_set_voice_file(
                state=state,
                tts_type=TtsModelInfos.INDEXTTS2,
                is_secondary=True,
                message_override="Enter emotion reference audio clip file path:"
            )

        # 5
        def on_clear_emo(_, __) -> None:
            project.clear_voice_and_save(TtsModelInfos.INDEXTTS2, is_secondary=True)
            printt_set("Cleared")

        # 6
        def make_vector_label(_) -> str:
            if project.indextts2_emo_vector:
                current = make_currently_string(project.emo_vector_to_string())
            else:
                current = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Emotion vector {current}"

        # 7
        def make_emo_alpha_label(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(
                project.indextts2_emo_alpha,
                IndexTts2Protocol.DEFAULT_EMO_VOICE_ALPHA, 2
            )
            return f"Emotion alpha (strength) {make_currently_string(value)}"

        def on_emo_alpha(_, __) -> None:
            VoiceSubmenuShared.ask_number(
                project,
                "Enter emotion alpha (0 to 1.0):",
                0.1, 1.0,
                "indextts2_emo_alpha",
                "Emotion alpha set to:"
            )

        # 8
        def make_fp16_item(_) -> str:
            value = VoiceSubmenuShared.make_parameter_value_string(project.indextts2_use_fp16, IndexTts2Protocol.DEFAULT_USE_FP16)
            return f"Toggle FP16 (smaller memory footprint) {make_currently_string(value)}"

        def on_fp16(_, __) -> None:
            project.indextts2_use_fp16 = not project.indextts2_use_fp16
            project.save()
            Tts.set_model_params_using_project(project)
            printt_set(f"FP16 set to: {project.indextts2_use_fp16}")

        # Menu
        items = [
            MenuItem(make_voice_label, on_voice),
            MenuItem("Clear voice clone sample", on_clear_voice),
            MenuItem(make_temperature_label, on_temperature),
            MenuItem(make_emo_voice_label, on_emo_voice),
            MenuItem("Clear emotion voice sample", on_clear_emo),
            MenuItem(make_vector_label, lambda _, __: ask_vector(project)),
            MenuItem(make_emo_alpha_label, on_emo_alpha),
            MenuItem(make_fp16_item, on_fp16)
        ]
        MenuUtil.menu(state, VoiceSubmenuShared.MENU_HEADING, items)

# ---

def ask_vector(project: Project) -> None:

    printt("Enter emotion vector:")
    printt()
    s = "This is a list of eight numbers between 0-1 corresponding to:\n"
    s += "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm\n"
    s += f'{COL_DIM}Eg: "0, 0.8, 0, 0, 0.2, 0, 0, 0" = very angry, slightly disgusted{COL_DEFAULT}'
    printt(s)
    printt()
    inp = ask("")
    if not inp:
        return
    result = Project.parse_emo_vector_string(inp)
    if isinstance(result, str):
        ask_error(result)
    else:
        project.indextts2_emo_vector = result
        project.save()
        printt_set(f"Emotion vector saved: {project.emo_vector_to_string()}")
