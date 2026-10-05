from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_hints import HINT_INDEX_SAMPLE_LEN
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.indextts2_base_model import IndexTts2BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DEFAULT, COL_DIM, COL_ERROR, make_currently_string, print_feedback, printt
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceIndexTts2Menu:

    @staticmethod
    def menu(state: State) -> None:
        project = state.project

        def voice_label(_) -> str:
            if project.get_model_setting('indextts2_local', 'file_name'):
                currently = make_currently_string(ProjectVoiceUtil.get_voice_label(project))
            else:
                currently = f"{COL_DIM}({COL_ERROR}required{COL_DIM})" # nb
            return f"Select voice clone sample {currently}"

        def make_emo_voice_label(_) -> str:
            if project.get_model_setting('indextts2_local', 'emo_voice'):
                value = ProjectVoiceUtil.make_voice_sample_display_label(
                    project, project.get_model_setting('indextts2_local', 'emo_voice'), TtsModelType.require_by_id("indextts2_local").value
                )
                currently = make_currently_string(value)
            else:
                currently = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Select emotion voice sample {currently}"

        def on_emo_voice(_: State, __: MenuItem) -> None:
            # TODO: disallow emo voice file == voice file (bc is default behavior anyway)
            hints.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN)
            VoiceMenuShared.ask_and_set_voice_file(
                state=state,
                tts_type=TtsModelType.require_by_id("indextts2_local"),
                is_secondary=True,
                message_override="Enter emotion reference audio clip file path:"
            )

        def on_clear_emo(_: State, __: MenuItem) -> None:
            ProjectVoiceUtil.clear_voice_and_save(project, TtsModelType.require_by_id("indextts2_local"), is_secondary=True)
            print_feedback("Cleared")

        def make_vector_label(_) -> str:
            if project.get_model_setting('indextts2_local', 'emo_vector'):
                current = make_currently_string(ProjectVoiceUtil.emo_vector_to_string(project))
            else:
                current = f"{COL_DIM}(optional){COL_DEFAULT}"
            return f"Emotion vector {current}"

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("indextts2_local"),
                no_samples_label=voice_label,
                on_before_set_callback=lambda: hints.show_hint_if_necessary(state.prefs, HINT_INDEX_SAMPLE_LEN),
            )
            items.append(
                MenuItem(make_emo_voice_label, on_emo_voice, superlabel="Emotion")
            )
            if project.get_model_setting('indextts2_local', 'emo_voice'):
                items.append(
                    MenuItem("Clear emotion voice sample", on_clear_emo)
                )
            items.append(
                MenuItem(make_vector_label, lambda _, __: ask_vector(project))
            )
            items.append(
                MenuUtil.make_number_item(
                    state=state,
                    target=SettingRef("indextts2_local", "emo_alpha"),
                    base_label="Emotion alpha (strength)",
                    default_value=IndexTts2BaseModel.DEFAULT_EMO_VOICE_ALPHA,
                    is_minus_one_default=True,
                    num_decimals=2,
                    prompt=f"Enter emotion alpha {COL_DIM}({0.01} to {1.0}){COL_DEFAULT}:",
                    min_value=0.01,
                    max_value=1.0
                )
            )
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


def ask_vector(project: Project) -> None:

    printt("Enter emotion vector:")
    printt()
    s = "This should be a list of eight numbers between 0-1 corresponding to:\n"
    s += "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm\n"
    s += f'{COL_DIM}Eg: "0, 0.8, 0, 0, 0.2, 0, 0, 0" = very angry, slightly disgusted{COL_DEFAULT}\n'
    s += f'{COL_DIM}Enter \"none\" to clear{COL_DEFAULT}'
    printt(s)
    printt()
    inp = ask.ask_input("")
    if not inp:
        return
    if inp == "none":
        value = []
    else:
        value = ProjectUtil.parse_emo_vector_string(inp)
        if isinstance(value, str):
            ask.ask_error(value)
            return
    if value == project.get_model_setting('indextts2_local', 'emo_vector'):
        return
    project.set_model_setting('indextts2_local', 'emo_vector', value)
    project.save()
    print_feedback("Emotion vector set:", ProjectVoiceUtil.emo_vector_to_string(project))
