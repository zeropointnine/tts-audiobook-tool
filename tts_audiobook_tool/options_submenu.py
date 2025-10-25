from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *

class OptionsSubmenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_whisper_label(_) -> str:
            value = make_currently_string(state.prefs.stt_variant.id)
            return f"Whisper transcription model {value}"

        whisper_item = MenuItem(
            make_whisper_label,
            lambda _, __: OptionsSubmenu.transcription_model_submenu(state)
        )

        def make_section_break_label(_) -> str:
            value = make_currently_string(str(state.prefs.use_section_sound_effect))
            return f"Use page turn sound effect at section breaks {value}"

        def on_section_break(_, __) -> None:
            AppUtil.show_hint_if_necessary(state.prefs, HINT_SECTION_SOUND_EFFECT)
            state.prefs.use_section_sound_effect = not state.prefs.use_section_sound_effect
            printt_set(f"Section break sound effect has been toggled to: {state.prefs.use_section_sound_effect}")

        section_break_item = MenuItem(make_section_break_label, on_section_break)


        def make_unload_label(_) -> str:
            result = AppUtil.get_nv_vram()
            if result is None:
                vram_label = ""
            else:
                used, total = result
                vram_label = f"{COL_DIM}(system VRAM: {make_gb_string(used)}/{make_gb_string(total)})"
            return f"Attempt to unload models {vram_label}"

        def on_unload(_, __) -> None:
            result = Tts.clear_all_models()
            if result:
                message = f"VRAM usage before: {make_gb_string(result[0])}\n"
                message += f"VRAM usage after: {make_gb_string(result[1])}"
            else:
                message = "OK"
            printt_set(message)

        unload_item = MenuItem(make_unload_label, on_unload)


        def on_hints(_, __) -> None:
            state.prefs.reset_hints()
            s = "One-time contextual hints have been reset.\n"
            s += "They will now appear again when relevant."
            printt_set(s)

        hints_item = MenuItem("Reset contextual hints", on_hints)


        items = [ whisper_item, section_break_item, unload_item, hints_item ]
        MenuUtil.menu(state, "Options:", items)


    @staticmethod
    def transcription_model_submenu(state: State) -> None:

        def make_heading(_) -> str:
            value = make_currently_string(state.prefs.stt_variant.value[0])
            return f"Select Whisper transcription model {value}"

        def handler(_, data: Any) -> None:
            if not data or not isinstance(data, SttVariant):
                return
            state.prefs.stt_variant = data
            Stt.set_variant(state.prefs.stt_variant) # sync global value
            printt_set(f"Whisper transcription model set to: {state.prefs.stt_variant.id}")

        menu_items = []
        for i, stt_variant in enumerate(SttVariant):
            label = f"{stt_variant.id} {COL_DIM}({stt_variant.description})"
            menu_item = MenuItem(label, handler, data=stt_variant)
            menu_items.append(menu_item)

        MenuUtil.menu(state, make_heading, menu_items)
