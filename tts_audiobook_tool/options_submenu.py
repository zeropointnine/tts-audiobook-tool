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

        def make_section_break_label(_) -> str:
            value = make_currently_string(str(state.prefs.use_section_sound_effect))
            return f"Insert page turn sound effect at section breaks {value}"

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
            print_feedback(message)

        def on_hints(_, __) -> None:
            state.prefs.reset_hints()
            s = "One-time contextual hints have been reset.\n"
            s += "They will now appear again when relevant."
            print_feedback(s)

        items = [
            MenuItem(
                make_whisper_label,
                lambda _, __: OptionsSubmenu.transcription_model_menu(state)
            ),
            MenuItem(
                make_section_break_label,
                lambda _, __: OptionsSubmenu.section_break_menu(state)
            ),
            MenuItem(make_unload_label, on_unload),
            MenuItem("Reset contextual hints", on_hints)
        ]
        MenuUtil.menu(state, "Options:", items)


    @staticmethod
    def transcription_model_menu(state: State) -> None:

        def make_heading(_) -> str:
            value = make_currently_string(state.prefs.stt_variant.value[0])
            return f"Select Whisper transcription model {value}"

        def handler(_, item: MenuItem) -> bool:
            if not item.data or not isinstance(item.data, SttVariant):
                return False
            state.prefs.stt_variant = item.data
            Stt.set_variant(state.prefs.stt_variant) # sync global value
            print_feedback(f"Whisper transcription model set to:", state.prefs.stt_variant.id)
            return True

        menu_items = []
        for i, stt_variant in enumerate(SttVariant):
            label = f"{stt_variant.id} {COL_DIM}({stt_variant.description})"
            menu_item = MenuItem(label, handler, data=stt_variant)
            menu_items.append(menu_item)

        MenuUtil.menu(state, make_heading, menu_items)

    @staticmethod
    def section_break_menu(state: State) -> None:

        def on_item(_, item: MenuItem) -> bool:
            state.prefs.use_section_sound_effect = item.data
            print_feedback(
                "Section break sound effect has been set to:",
                str(state.prefs.use_section_sound_effect)
            )
            return True

        items = [
            MenuItem("True", on_item, data=True),
            MenuItem("False", on_item, data=False)
        ]
        MenuUtil.menu(
            state,
            "Insert page turn sound effect at section breaks",
            items,
            hint=HINT_SECTION_SOUND_EFFECT,
            one_shot=True
        )
