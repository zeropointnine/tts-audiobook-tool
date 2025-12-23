from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import GlmProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu_shared import VoiceMenuShared

class VoiceGlmMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:

            items = [
                MenuItem(
                    VoiceMenuShared.make_select_voice_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.GLM)
                )
            ]

            if state.project.glm_voice_file_name:
                items.append( VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.GLM) )

            seed_value = str(state.project.glm_seed) if state.project.glm_seed > -1 else "random"
            items.extend([
                MenuItem(
                    make_menu_label("Model samplerate", str(state.project.glm_sr) + "hz"),
                    lambda _, __: samplerate_menu(state)
                ),
                MenuItem(
                    make_menu_label("Seed", seed_value),
                    lambda _, __: ask_seed(state)                    
                )
            ])
            return items
        
        VoiceMenuShared.show_voice_menu(state, make_items)

def samplerate_menu(state: State) -> None:

    def on_select(value: int) -> None:
        state.project.glm_sr = value
        state.project.save()
        Tts.set_model_params_using_project(state.project)
        print_feedback(f"Model samplerate set to:", str(value))        

    model_name = Tts.get_type().value.ui['short_name']

    MenuUtil.options_menu(
        state=state,
        heading_text="GLM model samplerate",
        labels=[str(item) + "hz" for item in GlmProtocol.SAMPLE_RATES],
        values=GlmProtocol.SAMPLE_RATES,
        current_value=state.project.glm_sr,
        default_value=GlmProtocol.SAMPLE_RATES[0],
        on_select=on_select
    )

def ask_seed(state: State) -> None:
    prompt = "Enter static seed value or -1 for random: "
    value = AskUtil.ask(prompt)
    if not value:
        return
    try:
        # fyi, always cast to float bc "int(5.1)"" throws exception in 3.11 seems like
        value = float(value)
    except Exception as e:
        print_feedback("Bad value", is_error=True)
        return
    value = int(value)
    if value < -1:
        print_feedback("Out of range", is_error=True)
        return

    state.project.glm_seed = value
    state.project.save()
    print_feedback("Seed set to:", value if value > -1 else "random")
