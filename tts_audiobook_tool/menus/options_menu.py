import torch
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.menus.llm_settings_menu import LlmSettingsMenu
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.models_util import ModelsUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class OptionsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_unload_label(_) -> str:
            memory_string = AppUtil.make_memory_string()
            if memory_string:
                memory_string = f"{COL_DIM}({memory_string}{COL_DIM})"
            return f"Attempt to unload models {memory_string}"

        def on_unload(_: State, __: MenuItem) -> None:
            before_string = AppUtil.make_memory_string()
            AppUtil.log_unload_memory_snapshot("before")
            if before_string:
                printt(f"Before: {before_string}")
            ModelsUtil.clear_all_models()
            after_string = AppUtil.make_memory_string()
            AppUtil.log_unload_memory_snapshot("after")
            if after_string:
                printt(f"After:  {after_string}")
            if state.prefs.menu_clears_screen:
                printt()
                ask.ask_enter_to_continue()

        def on_hints(_: State, __: MenuItem) -> None:
            state.prefs.reset_hints()
            s = "One-time contextual hints have been reset.\n"
            s += "They will now appear again when relevant."
            print_feedback(s)

        def item_maker(_: State) -> list[MenuItem]:

            items = []

            # Whisper model
            from tts_audiobook_tool.stt import Stt
            if not Stt.should_use_mlx_whisper():
                items.append(
                    MenuItem(
                        lambda _: make_menu_label("Whisper model", state.prefs.stt_variant.id, SttVariant.get_default().id),
                        lambda _, __: OptionsMenu.stt_model_menu(state),
                        superlabel="Model options", superlabel_no_blank_line=True
                    )
                )

            # Whisper device
            if not Stt.should_use_mlx_whisper():
                items.append(
                    MenuItem(
                        lambda _: make_menu_label("Whisper device", state.prefs.stt_config.description),
                        lambda _, __: OptionsMenu.whisper_device_menu(state)
                    )
                )

            # TTS force cpu
            import torch
            model_devices = Tts.get_type().value.torch_devices
            has_gpu = (
                (torch.cuda.is_available() and "cuda" in model_devices) or
                (torch.backends.mps.is_available() and "mps" in model_devices)
            )
            if model_devices and has_gpu:
                items.append(
                    MenuItem(
                        make_menu_label("TTS model - Force CPU", state.prefs.tts_force_cpu, False),
                        lambda _, __: OptionsMenu.tts_force_cpu_menu(state)
                    )
                )
                
            # About TTS model
            if Tts.get_type() != TtsModelInfos.NONE:
                items.append(
                    MenuItem(
                        lambda _: f"TTS model - About {Tts.get_type().value.ui['proper_name']}",
                        lambda _, __: print_about_model(state)
                    )
                )

            # Unload models
            items.append( MenuItem(make_unload_label, on_unload) )

            # Various:
            
            items.append(
                MenuItem(
                    lambda _: make_menu_label(
                        "LLM settings",
                        text_util.make_terminal_hyperlink(
                            state.prefs.llm_url,
                            ellipsize(state.prefs.llm_url, 50)
                        ) if state.prefs.llm_url else "none"
                    ),
                    lambda _, __: LlmSettingsMenu.menu(state),
                    superlabel="Various"
                )
            )
            items.append(
                MenuItem(
                    make_menu_label("AAC/M4B bitrate", state.prefs.aac_bitrate, AAC_BITRATE_DEFAULT),
                    lambda _, __: OptionsMenu.aac_bitrate_menu(state)
                )
            )
            items.append(
                MenuItem(
                    lambda _: make_menu_label("Menu clears screen", state.prefs.menu_clears_screen, False),
                    lambda _, __: OptionsMenu.menu_clears_screen_menu(state)
                )
            )
            items.append( MenuItem("Reset contextual hints", on_hints) )

            items.append( 
                MenuItem(
                    make_menu_label("Save debug files", state.prefs.save_debug_files, False),
                    lambda _, __: OptionsMenu.save_debug_files_menu(state)
                )
            )
            return items
        
        MenuUtil.menu(state, "Options:", item_maker, breadcrumb="Options")

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
            default_value=SttVariant.get_default(),
            on_select=on_select,
            hint=HINT_TRANSCRIPTION,
            breadcrumb="Whisper model",
        )

    @staticmethod
    def whisper_device_menu(state: State) -> None:

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
            on_select=on_select,
            breadcrumb="Whisper device",
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

    @staticmethod
    def save_debug_files_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.prefs.save_debug_files != value:
                state.prefs.save_debug_files = value
            print_feedback(f"Set to:", str(state.prefs.save_debug_files))

        subheading = f"Saves intermediate sound files alongside the \n"
        subheading += f"final sound segment FLAC files.\n"

        MenuUtil.options_menu(
            state=state,
            heading_text="Save debug files",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.save_debug_files,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def menu_clears_screen_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.prefs.menu_clears_screen != value:
                state.prefs.menu_clears_screen = value
            print_feedback(f"Set to:", str(state.prefs.menu_clears_screen))

        subheading = f"When enabled, clears screen between menus\n"
        subheading += f"and shows persistent status info.\n"

        MenuUtil.options_menu(
            state=state,
            heading_text="Menu clears screen",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.menu_clears_screen,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def aac_bitrate_menu(state: State) -> None:

        def on_select(value: str) -> None:
            if state.prefs.aac_bitrate != value:
                state.prefs.aac_bitrate = value
            print_feedback(f"Set AAC/M4B bitrate to:", state.prefs.aac_bitrate)

        MenuUtil.options_menu(
            state=state,
            heading_text="AAC/M4B bitrate",
            labels=AAC_BITRATES,
            values=AAC_BITRATES,
            current_value=state.prefs.aac_bitrate,
            default_value=AAC_BITRATE_DEFAULT,
            on_select=on_select
        )

# ---

def print_about_model(state: State) -> None:

    from tts_audiobook_tool.tts import Tts
    from tts_audiobook_tool import ask
    from tts_audiobook_tool.menus.menu_util import MenuUtil

    ui = Tts.get_type().value.ui
    model_name = ui["proper_name"]
    MenuUtil.print_screen_heading(state, f"About {model_name}")

    for link in ui.get("project_links", []):
        printt(text_util.make_terminal_hyperlink(link))
    printt(f"{COL_DIM}Use of this model is governed by the model's own license.")
    printt()
    ask.ask_enter_to_continue()



DEBUG_SUBHEADING = \
"""Saves intermediate sound segment files and diagnostic json data
alongside the regular sound segment FLAC files, and
preserves intermediate sound files after concatenation."""
