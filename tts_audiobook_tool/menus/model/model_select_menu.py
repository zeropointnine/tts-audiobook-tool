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
    def make_menu_label(state: State) -> str:
        """
        Menu item label, eg, `Switch TTS model (3 available) (currently: Chatterbox)`
        The `Switch` prefix and `available` count are only shown when there is a choice to make
        """
        num_available = len(Tts.get_available_tts_models())
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
            if len(Tts.get_available_tts_models()) >= 2:
                return f"{COL_ERROR}requires selection"
            return "None (unselected)"
        return model.value.ui["proper_name"]

    @staticmethod
    def menu(state: State) -> None:
        available = Tts.get_available_tts_models(refresh=True)
        models = [TtsModelType.require_by_id("none")] + sorted(
            dict.fromkeys(available), key=lambda model: model.value.ui["proper_name"].casefold(),
        )
        snapshot = RemoteTtsDiscovery.get_snapshot()

        def make_label(model: TtsModelType) -> str:
            label = model.value.ui["proper_name"]
            if (snapshot.backend_kind is not TtsBackendKind.AUDIO_CPP
                    or model.value.backend_kind is not TtsBackendKind.AUDIO_CPP):
                return label
            server_ids = [server_id for candidate, server_id in snapshot.candidates if candidate == model]
            if not server_ids:
                return label
            # Match binding's first-entry policy, even with several candidates.
            entries = [entry for entry in snapshot.models if entry.get("id") == server_ids[0]]
            if len(entries) != 1:
                return label
            path = entries[0].get("path")
            if isinstance(path, str):
                # The server may use either path style, independently of the client OS.
                filename = ntpath.basename(path.strip())
                if filename and filename not in (".", ".."):
                    label += f" {COL_DIM}({filename}){COL_DEFAULT}"
            return label

        labels = ["None (unselected)"] + [make_label(model) for model in models[1:]]

        def on_select(model: TtsModelType) -> None:
            state.pending_tts_model_change = None
            state.project.tts_model_type = model.id
            error = state.project.save()
            if error:
                ask.ask_error(error)
            Tts.bind_project(state.project)
            # Re-arm the model-dependent project FYIs (voice sample durations,
            # segmentation max words) so they re-check against the new model.
            state.pending_project_load_checks = True
            print_feedback("Project TTS model:", ModelSelectMenu.make_model_label(state))

        MenuUtil.options_menu(
            state=state, heading_text="TTS model", labels=labels,
            values=models, current_value=(state.project.get_tts_model_type()
                if state.project.tts_model_type == state.project.get_tts_model_type().id else None),
            default_value=None, on_select=on_select,
            breadcrumb="TTS model",
        )
