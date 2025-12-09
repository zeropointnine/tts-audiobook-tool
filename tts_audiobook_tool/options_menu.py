from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class OptionsSubmenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_whisper_model_label(_) -> str:
            value = make_currently_string(state.prefs.stt_variant.id)
            return f"Whisper model type {value}"

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
                    make_whisper_model_label,
                    lambda _, __: OptionsSubmenu.transcription_model_menu(state)
                ),
                MenuItem(
                    make_whisper_device_label,
                    lambda _, __: OptionsSubmenu.transcription_model_config_menu(state)
                )
            ]

            # TODO: add per-model protocol getter which can return None for OUTE; and use this getter in the Tts get_[model] functions
            b = Tts.get_best_torch_device() != "cpu" and Tts.get_type() != TtsModelInfos.OUTE 
            if b:
                item = MenuItem(make_tts_force_cpu_label, lambda _, __: OptionsSubmenu.tts_force_cpu_menu(state))
                items.append(item)

            items.extend([
                MenuItem(make_unload_label, on_unload),
                MenuItem(
                    make_section_break_label,
                    lambda _, __: OptionsSubmenu.section_break_menu(state)
                ),
                MenuItem("Reset contextual hints", on_hints)
            ])
            return items
        
        MenuUtil.menu(state, "Options:", item_maker)


    @staticmethod
    def transcription_model_menu(state: State) -> None:

        def make_heading(_) -> str:
            value = make_currently_string(state.prefs.stt_variant.value[0])
            return f"Whisper model type {value}"

        def handler(_: State, item: MenuItem) -> bool:
            if not item.data or not isinstance(item.data, SttVariant):
                return False
            state.prefs.stt_variant = item.data
            Stt.set_variant(state.prefs.stt_variant) # sync static value
            print_feedback(f"Set to:", state.prefs.stt_variant.id)
            return True

        menu_items = []
        for stt_variant in SttVariant:
            label = f"{stt_variant.id} {COL_DIM}({stt_variant.description})"
            menu_item = MenuItem(label, handler, data=stt_variant)
            menu_items.append(menu_item)

        MenuUtil.menu(state, make_heading, menu_items, one_shot=True)

    @staticmethod
    def transcription_model_config_menu(state: State) -> None:

        def on_item(_: State, item: MenuItem) -> bool:
            stt_config: SttConfig = item.data
            if state.prefs.stt_config != stt_config:
                state.prefs.stt_config= stt_config
                Stt.set_config(state.prefs.stt_config) # sync static value
            print_feedback(f"Set to:", str(state.prefs.stt_config.description))
            return True

        items = []
        for dq in list(SttConfig):
            items.append( MenuItem(dq.description, on_item, data=dq) )

        MenuUtil.menu(
            state,
            make_whisper_device_label(state),
            items,
            one_shot=True
        )

    @staticmethod
    def tts_force_cpu_menu(state: State) -> None:

        def handler(_: State, item: MenuItem) -> bool:            
            if state.prefs.tts_force_cpu != item.data:
                state.prefs.tts_force_cpu = item.data
                Tts.set_force_cpu(state.prefs.tts_force_cpu)
            print_feedback(f"Set to:", str(state.prefs.tts_force_cpu))
            return True

        items = []
        for value in [True, False]:
            label = f"{value}"
            if value == False:
                 label += f" {COL_DIM}(default)"
            menu_item = MenuItem(label, handler, data=value)
            items.append(menu_item)

        subheading = f"Forces the TTS model ({Tts.get_type().value.ui['short_name']}) to use cpu as its torch device\n" + \
            "even when CUDA or MPS is available.\n"

        MenuUtil.menu(
            state, 
            make_tts_force_cpu_label(state), 
            items, 
            subheading=subheading,
            one_shot=True
        )

    @staticmethod
    def section_break_menu(state: State) -> None:

        def on_item(_: State, item: MenuItem) -> bool:
            state.prefs.use_section_sound_effect = item.data
            print_feedback(
                "Set to:",
                str(state.prefs.use_section_sound_effect)
            )
            return True

        items = [
            MenuItem("True", on_item, data=True),
            MenuItem(f"False {COL_DIM}(default)", on_item, data=False)
        ]
        MenuUtil.menu(
            state,
            make_section_break_label,
            items,
            subheading=SECTION_BREAK_SUBHEADING,
            one_shot=True
        )

# ---

def make_whisper_device_label(state: State) -> str:
    value = make_currently_string(state.prefs.stt_config.description)
    return f"Whisper device {value}"

def make_tts_force_cpu_label(state: State) -> str:
    value = make_currently_string(state.prefs.tts_force_cpu)
    return f"TTS model - use CPU as device {value}"
        
def make_section_break_label(state: State) -> str:
    value = make_currently_string(state.prefs.use_section_sound_effect)
    return f"Insert sound effect at section breaks {value}"
        
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

SECTION_BREAK_SUBHEADING = \
"""In the concatenation step, inserts a page turn sound effect when 
two or more consecutive blank lines are encountered in the text. 
This can be a useful audible cue, so long as the text is formatted for it.
"""
