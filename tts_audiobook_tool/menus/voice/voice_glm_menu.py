from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.menus.voice import VoiceMenuShared

class VoiceGlmMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:

            items = []
            items.append(
                VoiceMenuShared.make_manage_voice_samples_item(state, TtsModelType.GLM)
            )
            items.append(
                MenuItem(
                    make_menu_label("Model samplerate", str(state.project.glm_sr) + "hz"),
                    lambda _, __: samplerate_menu(state)
                )
            )
            
            item = VoiceMenuShared.make_seed_item(state, "glm_seed")
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)
            
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)

def samplerate_menu(state: State) -> None:

    def on_select(value: int) -> None:
        state.project.glm_sr = value
        state.project.save()
        Tts.set_model_params_using_project(state.project)
        print_feedback(f"Model samplerate set to:", str(value))        

    MenuUtil.options_menu(
        state=state,
        heading_text="GLM model samplerate",
        labels=[str(item) + "hz" for item in GlmBaseModel.SAMPLE_RATES],
        values=GlmBaseModel.SAMPLE_RATES,
        current_value=state.project.glm_sr,
        default_value=GlmBaseModel.SAMPLE_RATES[0],
        on_select=on_select
    )
