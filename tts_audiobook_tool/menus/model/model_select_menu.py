"""Remote-mode TTS model picker, shown at the top of the Model settings menu."""
import ntpath

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *


class ModelSelectMenu:

    @staticmethod
    def make_item(state: State) -> MenuItem:
        return MenuItem(
            lambda _: ModelSelectMenu.make_menu_label(state),
            lambda _, __: ModelSelectMenu.menu(state),
        )

    @staticmethod
    def is_selection_required(state: State) -> bool:
        """Use the shared selection policy, including ambiguous legacy entries."""
        return Tts.is_project_model_selection_required(state.project)

    @staticmethod
    def _get_choices(available: list[TtsModelType]) -> list[tuple[TtsModelType, str]]:
        """Count exact audio.cpp entries, but retain distinct types elsewhere."""
        snapshot = RemoteTtsDiscovery.get_snapshot()
        choices = []
        for model in dict.fromkeys(available):
            entries = [candidate for candidate in snapshot.candidates if candidate[0] == model]
            if (model.value.backend_kind is TtsBackendKind.AUDIO_CPP
                    and snapshot.backend_kind is TtsBackendKind.AUDIO_CPP and entries):
                choices.extend(entries)
            else:
                choices.append((model, ""))
        return sorted(dict.fromkeys(choices), key=lambda choice: (
            choice[0].value.ui["proper_name"].casefold(), choice[1].casefold(), choice[1],
        ))

    @staticmethod
    def make_menu_label(state: State) -> str:
        """Show the number of selectable entries, not just catalog types."""
        num_available = len(ModelSelectMenu._get_choices(Tts.get_available_tts_models()))
        label = "TTS model"
        if num_available > 1:
            label = f"Switch TTS model {COL_DIM}({num_available} available)"
        return make_menu_label(label, ModelSelectMenu.make_model_label(state))

    @staticmethod
    def make_model_label(state: State) -> str:
        model = state.project.get_tts_model_type()
        if model.id == "none":
            if state.project.tts_model_type != "none":
                return f"Unknown model: {state.project.tts_model_type}"
            if ModelSelectMenu.is_selection_required(state):
                return f"{COL_ERROR}requires selection"
            return "None (unselected)"
        label = model.value.ui["proper_name"]
        if model.value.backend_kind is TtsBackendKind.AUDIO_CPP and state.project.tts_server_model_id:
            label += f" {COL_DIM}({state.project.tts_server_model_id}){COL_DEFAULT}"
        return label

    @staticmethod
    def menu(state: State) -> None:
        available = Tts.get_available_tts_models(refresh=True)
        choices = [(TtsModelType.require_by_id("none"), "")] + ModelSelectMenu._get_choices(available)
        snapshot = RemoteTtsDiscovery.get_snapshot()

        def make_label(choice: tuple[TtsModelType, str]) -> str:
            model, server_id = choice
            if model.id == "none":
                return "None (unselected)"
            label = model.value.ui["proper_name"]
            if not server_id:
                return label
            details = server_id
            entries = [entry for entry in snapshot.models if entry.get("id") == server_id]
            if len(entries) == 1:
                path = entries[0].get("path")
                if isinstance(path, str):
                    # Server path style is independent of the client's OS.
                    filename = ntpath.basename(path.strip())
                    if filename and filename not in (".", ".."):
                        details += f"; {filename}"
            return f"{label} {COL_DIM}({details}){COL_DEFAULT}"

        def on_select(choice: tuple[TtsModelType, str]) -> None:
            model, server_id = choice
            state.pending_tts_model_change = None
            state.project.tts_model_type = model.id
            state.project.tts_server_model_id = server_id
            error = state.project.save()
            if error:
                ask.ask_error(error)
            Tts.bind_project(state.project)
            # Re-arm model-dependent project FYIs after an explicit selection.
            state.pending_project_load_checks = True
            print_feedback("Project TTS model:", ModelSelectMenu.make_model_label(state))

        model = state.project.get_tts_model_type()
        current = ((model, state.project.tts_server_model_id
                    if model.value.backend_kind is TtsBackendKind.AUDIO_CPP else "")
                   if state.project.tts_model_type == model.id else None)
        MenuUtil.options_menu(
            state=state, heading_text="TTS model", labels=[make_label(choice) for choice in choices],
            values=choices, current_value=current,
            default_value=None, on_select=on_select,
            breadcrumb="TTS model",
        )
