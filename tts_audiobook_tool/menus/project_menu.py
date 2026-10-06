from dataclasses import replace
import ntpath

from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.menus.project_new_menu import ProjectNewMenu
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.system_support.platforms import open_directory
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.text_ops.whitelist import Whitelist

class ProjectMenu:

    @staticmethod
    def menu(state:State) -> None:

        def make_heading(_: State) -> str:
            return "Project"

        def on_new_project(_: State, __: MenuItem) -> None:
            ProjectNewMenu.menu(state)

        def on_existing_project(_: State, __: MenuItem) -> None:
            did = ProjectMenu.ask_and_set_existing_project(state)
            if did:
                print_feedback("Project directory set:", state.project.dir_path)

        def on_view(_: State, __: MenuItem) -> None:
            err = open_directory(state.project.dir_path)
            if err:
                ask.ask_error(err)
            else:
                print_feedback("Launched window")

        def on_clear_language(_: State, __: MenuItem) -> None:
            state.project.language_code = ""
            Whitelist().set_language_code("")
            state.project.save()
            print_feedback("Language code cleared")

        def items_maker(_) -> list[MenuItem]:

            items = []

            items.append(
                MenuItem("New project", on_new_project, data=True)
            )

            items.append(
                MenuItem("Open existing project", on_existing_project, data=False)
            )

            if state.project.dir_path:

                remote_mode = Tts.is_remote_mode()
                if remote_mode:
                    items.append(MenuItem(
                        lambda _: ProjectMenu.make_tts_model_menu_label(state),
                        lambda _, __: ProjectMenu.tts_model_menu(state),
                        superlabel="Options",
                    ))
                items.append(
                    MenuItem(
                        lambda _: make_menu_label("Language code", state.project.language_code or "none"),
                        on_language,
                        superlabel="" if remote_mode else "Options",
                    )
                )

                if state.project.language_code:
                    items.append(
                        MenuItem("Clear language code", on_clear_language)
                    )

                items.append(
                    MenuItem(
                        "Open project directory in system file explorer", on_view,
                        superlabel=" ", superlabel_no_blank_line=True
                    )
                )

            return items

        MenuUtil.menu(
            state,
            make_heading,
            items_maker,
            breadcrumb="Project",
        )

    @staticmethod
    def make_tts_model_menu_label(state: State) -> str:
        """
        Menu item label, eg, `Switch TTS model (3 available) (currently: Chatterbox)`
        The `Switch` prefix and `available` count are only shown when there is a choice to make
        """
        num_available = len(Tts.get_available_tts_models())
        label = "TTS model"
        if num_available > 1:
            label = f"Switch TTS model {COL_DIM}({num_available} available)"
        return make_menu_label(label, ProjectMenu.make_tts_model_label(state))

    @staticmethod
    def make_tts_model_label(state: State) -> str:
        model = state.project.get_tts_model_type()
        if model.id == "none":
            if state.project.tts_model_type != "none":
                return f"Unknown model: {state.project.tts_model_type}"
            if len(Tts.get_available_tts_models()) >= 2:
                return f"{COL_ERROR}requires selection"
            return "None (unselected)"
        return model.value.ui["proper_name"]

    @staticmethod
    def tts_model_menu(state: State) -> None:
        available = Tts.get_available_tts_models(refresh=True)
        models = [TtsModelType.require_by_id("none")] + list(dict.fromkeys(available))
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
            print_feedback("Project TTS model:", ProjectMenu.make_tts_model_label(state))

        MenuUtil.options_menu(
            state=state, heading_text="TTS model", labels=labels,
            values=models, current_value=(state.project.get_tts_model_type()
                if state.project.tts_model_type == state.project.get_tts_model_type().id else None),
            default_value=None, on_select=on_select,
            breadcrumb="TTS model",
        )

    @staticmethod
    def ask_and_set_existing_project(state: State) -> bool:
        """
        Asks user for directory and if valid, sets state to existing project
        Returns True on success
        """
        s = "Enter existing project directory path:"
        s2 = "Select existing project directory"
        dir = ask.ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=True)
        if not dir:
            return False
        err = ProjectLoadUtil.is_valid_project_dir(dir)
        if err:
            ask.ask_error(err)
            return False

        state.set_existing_project(dir)

        return True

# ---

def on_language(state: State, __: MenuItem) -> None:

    MenuUtil.print_screen_heading(state, "Language code", breadcrumb="Language code")
    printt(LANGUAGE_CODE_DESC)
    printt()

    # TODO: consider making this a property of TtsBaseModel
    required_model_languages = []

    # Chatterbox Multilingual special case
    if state.project.get_tts_model_type().id == "chatterbox_local" and state.project.get_model_setting('chatterbox_local', 'type').is_multilingual:
        inspection, error = ModelWorker.inspect_tts_blocking(state)
        if error or inspection is None:
            ask.ask_error(error or "Couldn't inspect Chatterbox model")
            return
        languages = (inspection.metadata or {}).get("supported_languages_multi", [])
        if isinstance(languages, (list, tuple)):
            required_model_languages = [str(value) for value in languages]
        printt(f"Chatterbox-Multilingual requires one of the following language codes:\n{required_model_languages}")
        printt()

    def validator(code: str) -> str:

        # (1) Super-basic syntax check
        code = code.strip()
        bad = len(code) > 5
        bad = bad or not any(char.isalpha() for char in code)
        if bad:
            return "Bad value"

        # (2) Required model language
        if required_model_languages and not code in required_model_languages:
            return "Language code not supported by Chatterbox Multilingual"

        # (3) Hint-side-effect re: CJK
        if Validator.is_unsupported_language_code(code): # (not to be confused with chatterbox multilingual requirement)
            text = HINT_VALIDATION_UNSUPPORTED_LANGUAGE.text.replace("%1", str(VALIDATION_UNSUPPORTED_LANGUAGES))
            hint = replace(HINT_VALIDATION_UNSUPPORTED_LANGUAGE, text=text)
            hints.show_hint(hint, and_prompt=True)

        # (4) Hint-side-effect re: strictness non-en
        if not Whitelist.supports_language(code) and state.project.strictness != Strictness.LOW:
            if not Validator.is_unsupported_language_code(code):
                state.project.strictness = Strictness.LOW
                state.project.save()
                hints.show_hint(HINT_FORCED_STRICTNESS_LOW, and_prompt=True)

        return ""

    prompt = f"Enter two-letter language code {COL_DIM}(Eg, \"en\", \"es\", \"zh\", \"pt\", etc){COL_DEFAULT}:"
    did_save = ask.ask_string_and_save(
        state.project,
        prompt,
        "language_code",
        "Project language code set to:",
        validator=validator,
        normalizer=lambda value: value.strip().lower(),
    )
    Whitelist().set_language_code(state.project.language_code)
    if did_save and Whitelist.supports_language(state.project.language_code):
        hints.show_hint_if_necessary(
            state.prefs,
            HINT_TOLERANCE_FIRST_CLASS,
            and_prompt=True,
        )

LANGUAGE_CODE_DESC = "" + \
"""Language code is used by the app at various stages of the pipeline as a \"hint\" for:
- Semantic segmentation of imported text
- Prompt pre-processing
- Whisper transcription
- TTS inference (eg, Chatterbox, MOSS)"""
