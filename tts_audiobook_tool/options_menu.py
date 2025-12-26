import torch
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *

class OptionsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_unload_label(_) -> str:
            memory_string = make_system_memory_string()
            if memory_string:
                memory_string = f"{COL_DIM}({memory_string}{COL_DIM})"
            return f"Attempt to unload models {memory_string}"

        def on_unload(_: State, __: MenuItem) -> None:
            before_string = make_system_memory_string()
            if before_string:
                printt(f"Before: {before_string}")
            Tts.clear_all_models()
            after_string = make_system_memory_string()
            if after_string:
                printt(f"After:  {after_string}")
            printt()

        def on_hints(_: State, __: MenuItem) -> None:
            state.prefs.reset_hints()
            s = "One-time contextual hints have been reset.\n"
            s += "They will now appear again when relevant."
            print_feedback(s)

        def item_maker(_: State) -> list[MenuItem]:
            items = [
                MenuItem(
                    lambda _: make_menu_label("Whisper model type", state.prefs.stt_variant.id),
                    lambda _, __: OptionsMenu.stt_model_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label("Whisper device", state.prefs.stt_config.description),
                    lambda _, __: OptionsMenu.stt_config_menu(state)
                )
            ]
            if Tts.get_type().value.torch_devices:
                item = MenuItem(
                    lambda _: make_menu_label("TTS model - CPU override", state.prefs.tts_force_cpu), 
                    lambda _, __: OptionsMenu.tts_force_cpu_menu(state)
                )
                items.append(item)

            items.extend([
                MenuItem(make_unload_label, on_unload),
                MenuItem("Reset contextual hints", on_hints)
            ])
            return items
        
        MenuUtil.menu(state, "Options:", item_maker)

    @staticmethod
    def stt_model_menu(state: State) -> None:

        def on_select(value: SttVariant) -> None:
            state.prefs.stt_variant = value
            print_feedback(f"Set to:", state.prefs.stt_variant.id)

        MenuUtil.options_menu(
            state=state,
            heading_text="Whisper model",
            labels=[item.id for item in list(SttVariant)],
            sublabels=[item.description for item in list(SttVariant)],
            values=[item for item in list(SttVariant)],
            current_value=state.prefs.stt_variant,
            default_value=list(SttVariant)[0],
            on_select=on_select,
            hint=HINT_TRANSCRIPTION
        )

    @staticmethod
    def stt_config_menu(state: State) -> None:

        def on_select(value: SttConfig) -> None:
            if state.prefs.stt_config != value:
                state.prefs.stt_config= value
            print_feedback(f"Set whisper device to:", str(state.prefs.stt_config.description))

        labels = []
        for item in list(SttConfig):
            s = item.description
            if item == SttConfig.CUDA_FLOAT16 and not torch.cuda.is_available():
                s += f" {COL_DIM}(unavailable)"
            labels.append(s)

        MenuUtil.options_menu(
            state=state,
            heading_text="Whisper device",
            labels=labels,
            values=[item for item in list(SttConfig)],
            current_value=state.prefs.stt_config,
            default_value=None,
            on_select=on_select
        )

    @staticmethod
    def tts_force_cpu_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.prefs.tts_force_cpu != value:
                state.prefs.tts_force_cpu = value
            print_feedback(f"Set to:", str(state.prefs.tts_force_cpu))        

        subheading = f"Forces TTS model to use CPU as its torch device even when GPU is available.\n"

        MenuUtil.options_menu(
            state=state,
            heading_text="TTS model - use CPU as device",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.tts_force_cpu,
            default_value=False,
            on_select=on_select
        )

# ---

def make_system_memory_string(base_color=COL_DIM) -> str:

    result = AppUtil.get_nv_vram()
    if result is None:
        vram_string = ""
    else:
        used, total = result
        vram_string = f"{base_color}VRAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

    result = AppUtil.get_system_ram()
    if result is None:
        ram_string = ""
    else:
        used, total = result
        ram_string = f"{base_color}RAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

    if not vram_string and not ram_string:
        return ""
    elif not vram_string:
        return ram_string
    elif not ram_string:
        return vram_string
    else:
        return f"{vram_string}{base_color}, {ram_string}"
