from dataclasses import replace
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.project_new_menu import ProjectNewMenu
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.system_support.platforms import open_directory
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.text_ops.whitelist import Whitelist

class ProjectMenu:

    @staticmethod
    def menu(state:State) -> None:

        def make_heading(_: State) -> str:
            value = text_util.make_terminal_hyperlink(state.project.dir_path, is_file=True) if state.project.dir_path else "none"
            heading = make_menu_label("Project", value) if not state.prefs.menu_clears_screen else "Project"
            return heading

        def on_new_project(_: State, __: MenuItem) -> None:
            ProjectNewMenu.menu(state)

        def on_existing_project(_: State, __: MenuItem) -> bool:
            did = ProjectMenu.ask_and_set_existing_project(state)
            if did:
                print_feedback("Project directory set:", state.project.dir_path)
                return True
            else:
                return False

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

                items.append( 
                    MenuItem(
                        lambda _: make_menu_label("Language code", state.project.language_code or "none"), 
                        on_language,
                        superlabel="Options"
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

        # Show over-default-max-words hint
        max_count = ProjectBookUtil.get_book_segmentation_settings(state.project).max_words_per_segment
        # TODO: Not doing this for now: ... or PhraseGroup.get_max_num_words(state.project.phrase_groups)
        reco_range: tuple[int, int] = Tts.get_type().value.max_words_reco_range
        if max_count > reco_range[1] and Tts.get_type() != TtsModelType.NONE:
            message = HINT_MAX_WORDS_OVER_DEFAULT_MESSAGE
            message = message.replace("%1", str(max_count))
            reco_str = TtsModelType.recommended_range_string(Tts.get_type().value)
            message = message.replace("%2", reco_str)
            hint = Hint("", "FYI", message)
            hints.show_hint(hint, and_prompt=True)

        return True

# ---

def on_language(state: State, __: MenuItem) -> None:

    MenuUtil.print_screen_heading(state, "Language code", breadcrumb="Language code")
    printt(LANGUAGE_CODE_DESC)
    printt()

    # TODO: consider making this a property of TtsBaseModel
    required_model_languages = []

    # Chatterbox Multilingual special case
    if Tts.get_type() == TtsModelType.CHATTERBOX and state.project.chatterbox_type == ChatterboxType.MULTILINGUAL:
        instance = Tts.get_instance() # Force model instantiation
        assert(isinstance(instance, ChatterboxBaseModel))
        required_model_languages = instance.supported_languages_multi()
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
    if did_save and state.project.language_code in ("en", "es"):
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
- TTS inference (Chatterbox, MOSS)"""
