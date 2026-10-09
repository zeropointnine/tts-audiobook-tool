from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.ask_advanced import AskAdvanced
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.menus.llm_settings_menu import LlmSettingsMenu
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.system_support.gpu_caps_util import GpuCapsUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.util import *

class OptionsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_unload(_: State, __: MenuItem) -> None:
            # All local models live inside the worker process, so a dead
            # worker guarantees nothing is loaded; a live worker can be
            # queried for its resident model inventory.
            if not ModelWorker.is_alive():
                print_feedback("No models to unload")
                return
            snapshot, state_error = ModelWorker.get_model_state_blocking()
            if snapshot is not None and not snapshot.any_loaded:
                print_feedback("No models to unload")
                return
            # If the state query failed (eg, worker busy), fall through and
            # attempt the unload anyway.
            reset_error = state_error or ModelWorker.unload_models_blocking()
            if reset_error:
                ask.ask_error(reset_error)
            else:
                print_feedback("Models unloaded")

        def on_hints(_: State, __: MenuItem) -> None:
            state.prefs.reset_hints()
            state.prefs.save()
            s = "One-time contextual hints have been reset.\n"
            s += "They will now appear again when relevant."
            print_feedback(s, long_pause=True)

        def item_maker(_: State) -> list[MenuItem]:

            items = []
            remote_mode = Tts.is_remote_mode()

            # Whisper controls are shared, but follow the server controls in remote mode.
            from tts_audiobook_tool.stt import Stt
            use_mlx_whisper = Stt.should_use_mlx_whisper()
            whisper_items = []
            if not use_mlx_whisper:
                whisper_items.append(MenuItem(
                    lambda _: make_menu_label("Whisper model", state.prefs.stt_variant.id, SttVariant.get_default().id),
                    lambda _, __: OptionsMenu.stt_model_menu(state),
                    superlabel="Speech-to-text model" if remote_mode else "Model options",
                    superlabel_no_blank_line=not remote_mode,
                ))
                whisper_items.append(MenuItem(
                    lambda _: make_menu_label("Whisper device", state.prefs.stt_config.description),
                    lambda _, __: OptionsMenu.whisper_device_menu(state),
                ))

            if remote_mode:
                items.append(MenuItem(
                    lambda _: make_menu_label(
                        "Server URL",
                        text_util.make_terminal_hyperlink(state.prefs.remote_tts_url,
                            ellipsize(state.prefs.remote_tts_url, 50))
                        if state.prefs.remote_tts_url else f"{COL_ERROR}required"),
                    lambda _, __: OptionsMenu.ask_remote_tts_url(state),
                    superlabel=lambda _: OptionsMenu.make_remote_tts_superlabel(),
                    superlabel_no_blank_line=True))
                items.append(MenuItem(
                    lambda _: OptionsMenu.make_refresh_remote_tts_label(),
                    lambda _, __: OptionsMenu.refresh_remote_tts(state)))
                if state.project.get_tts_model_type().id != "none":
                    items.append(MenuItem(
                        lambda _: f"About current TTS model: {state.project.get_tts_model_type().value.ui['proper_name']}",
                        lambda _, __: print_about_model(state),
                    ))
                snapshot = RemoteTtsDiscovery.get_snapshot()
                if (snapshot.backend_kind is TtsBackendKind.AUDIO_CPP
                        and any(model.get("loaded") is True for model in snapshot.models)):
                    items.append(MenuItem(
                        "Unload audio.cpp models",
                        lambda _, __: OptionsMenu.unload_audio_cpp_models(),
                    ))
                items.extend(whisper_items)
            else:
                items.extend(whisper_items)

                # TTS force cpu
                import torch
                from tts_audiobook_tool.app_types import DeviceType
                model_devices = state.project.get_tts_model_type().value.local_torch_devices
                has_gpu = (
                    (torch.cuda.is_available() and DeviceType.CUDA in model_devices) or
                    (torch.backends.mps.is_available() and DeviceType.MPS in model_devices)
                )
                if model_devices and has_gpu:
                    items.append(MenuItem(
                        make_menu_label("TTS model - Force CPU", state.prefs.tts_force_cpu, False),
                        lambda _, __: OptionsMenu.tts_force_cpu_menu(state),
                    ))

                # About TTS model
                if state.project.get_tts_model_type().id != "none":
                    items.append(MenuItem(
                        lambda _: f"TTS model - About {state.project.get_tts_model_type().value.ui['proper_name']}",
                        lambda _, __: print_about_model(state),
                    ))

            # Unload models
            items.append(MenuItem(
                "Unload local models", on_unload,
                superlabel="Speech-to-text model" if remote_mode and use_mlx_whisper else "",
            ))

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

            items.append( MenuItem("Reset contextual hints", on_hints) )

            items.append(
                MenuItem(
                    make_menu_label("Save log files", state.prefs.save_gen_log),
                    lambda _, __: OptionsMenu.save_gen_log_menu(state)
                )
            )

            items.append(
                MenuItem(
                    make_menu_label("Save debug files", state.prefs.save_debug_files),
                    lambda _, __: OptionsMenu.save_debug_files_menu(state)
                )
            )
            return items

        # Entering Options is an explicit server-inspection action: refresh
        # before the heading/status and controls are rendered, not on redraws.
        if Tts.is_remote_mode():
            RemoteTtsDiscovery.refresh(force=True)
        MenuUtil.menu(state, "Options", item_maker, breadcrumb="Options")

    @staticmethod
    def stt_model_menu(state: State) -> None:

        def on_select(value: SttVariant) -> None:
            state.prefs.stt_variant = value
            state.prefs.save()
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
                state.prefs.save()
            print_feedback(f"Set whisper device to:", str(state.prefs.stt_config.description))

        labels = []
        for item in list(SttConfig):
            s = item.description
            if item == SttConfig.CUDA_FLOAT16 and not GpuCapsUtil.has_ctranslate2_float16_gpu():
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
                state.prefs.save()
            print_feedback(f"Set to:", str(state.prefs.tts_force_cpu))

        subheading = f"Forces TTS model to use CPU as its torch device even when GPU is available."

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
    def save_gen_log_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.prefs.save_gen_log != value:
                state.prefs.save_gen_log = value
                state.prefs.save()
            print_feedback(f"Set to:", str(state.prefs.save_gen_log))

        subheading = "Saves log file when generating TTS audio"

        MenuUtil.options_menu(
            state=state,
            heading_text="Save log files",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.save_gen_log,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def save_debug_files_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if state.prefs.save_debug_files != value:
                state.prefs.save_debug_files = value
                state.prefs.save()
            print_feedback(f"Set to:", str(state.prefs.save_debug_files))

        segments_dir = text_util.make_terminal_hyperlink(
            state.project.sound_segments_path, PROJECT_SOUND_SEGMENTS_SUBDIR, is_file=True
        )
        subheading = (
            f"Saves intermediate sound files alongside finalized sound segment\n"
            f"FLAC files in the project {segments_dir} directory."
        )

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
    def aac_bitrate_menu(state: State) -> None:

        def on_select(value: str) -> None:
            if state.prefs.aac_bitrate != value:
                state.prefs.aac_bitrate = value
                state.prefs.save()
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

    @staticmethod
    def make_remote_tts_superlabel() -> str:
        backend = RemoteTtsDiscovery.get_snapshot().backend_kind
        backend_name = {
            TtsBackendKind.AUDIO_CPP: "audio.cpp",
            TtsBackendKind.SGL_OMNI: "SGL-Omni",
        }.get(backend) if backend is not None else None
        return f"Remote TTS server ({backend_name})" if backend_name else "Remote TTS server"

    @staticmethod
    def make_refresh_remote_tts_label() -> str:
        label = "Refresh server info"
        snapshot = RemoteTtsDiscovery.get_snapshot()
        if snapshot.backend_kind is TtsBackendKind.AUDIO_CPP:
            count = len(snapshot.candidates)
            noun = "model" if count == 1 else "models"
            return make_menu_label(label, f"{count} supported {noun} available")
        return label

    @staticmethod
    def unload_audio_cpp_models() -> None:
        print_feedback("Unloading...", skip_pause=True)
        error = AudioCppUtil.unload_all_models(RemoteTtsDiscovery.get_base_url())
        if error is not None:
            ask.ask_error(error)
            return
        RemoteTtsDiscovery.refresh(force=True)
        print_feedback("Successfully unloaded models")

    @staticmethod
    def refresh_remote_tts(state: State) -> None:
        snapshot = RemoteTtsDiscovery.refresh(force=True)
        Tts.bind_project(state.project)
        if snapshot.issue is not None:
            ask.ask_error(f"Remote TTS server:\n{snapshot.issue.message}")
        else:
            count = len(snapshot.candidates)
            noun = make_noun("model", "models", count)
            value = f"{snapshot.backend_kind.value if snapshot.backend_kind else 'unknown'}, {count} supported {noun}"
            print_feedback("Remote TTS server refreshed:", value, long_pause=True)

    @staticmethod
    def ask_remote_tts_url(state: State) -> None:
        printt(f"Enter Remote TTS server URL:\n{COL_DIM}(Eg, http://localhost:8000)")
        value = AskAdvanced.ask(prefill=state.prefs.remote_tts_url).strip()
        printt()
        if not value:
            return
        if "://" not in value:
            value = f"http://{value}"
        issue = RemoteTtsDiscovery.validate_url(value)
        if issue is not None:
            ask.ask_error(issue.message)
            return
        state.prefs.remote_tts_url = value.rstrip("/")
        state.prefs.save()
        Tts.bind_project(state.project)
        print_feedback("Set Remote TTS URL to:", state.prefs.remote_tts_url)

# ---

def print_about_model(state: State) -> None:

    from tts_audiobook_tool import ask
    from tts_audiobook_tool.menus.menu_util import MenuUtil

    ui = state.project.get_tts_model_type().value.ui
    model_name = ui["proper_name"]
    MenuUtil.print_screen_heading(state, f"About {model_name}")

    for link in ui.get("project_links", []):
        printt(text_util.make_terminal_hyperlink(link))
    printt()
    printt(f"{COL_DIM}Use of this model is governed by the model's own license.")
    printt()
    ask.ask_enter_to_continue()



DEBUG_SUBHEADING = \
"""Saves intermediate sound segment files and diagnostic json data
alongside the regular sound segment FLAC files, and
preserves intermediate sound files after concatenation."""
