from dataclasses import replace
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.dir_open_util import DirOpenUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.voice_menu_shared import VoiceMenuShared

class ProjectMenu:

    @staticmethod
    def menu(state:State) -> None:

        def on_new_project(_, menu_item) -> bool:            
            
            did = ProjectMenu.ask_and_set_new_project(state)
            if not did:
                return False

            print_feedback("Project directory set:", state.project.dir_path)
            AppUtil.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS, and_prompt=True)
            return True

        def on_existing_project(_: State, __: MenuItem) -> bool:
            did = ProjectMenu.ask_and_set_existing_project(state)
            if did:
                print_feedback("Project directory set:", state.project.dir_path)
                return True
            else:
                return False

        def on_view(_: State, __: MenuItem) -> None:
            err = DirOpenUtil.open(state.project.dir_path)
            if err:
                AskUtil.ask_error(err)
            else:
                print_feedback("Launched window")

        def on_clear_language(_: State, __: MenuItem) -> None:
            state.project.language_code = ""
            state.project.save()
            print_feedback("Language code cleared")            

        def items_maker(_) -> list[MenuItem]:
            items = [
                MenuItem("New project", on_new_project, data=True),
                MenuItem("Open existing project", on_existing_project, data=False),
                MenuItem(
                    lambda _: make_menu_label("Language code", state.project.language_code or "none"), 
                    on_language
                )
            ]
            if state.project.language_code:
                items.append(MenuItem("Clear language code", on_clear_language))
            if state.project.dir_path:
                items.append(MenuItem("Show directory in OS UI", on_view))
            return items

        MenuUtil.menu(
            state, 
            lambda _: make_menu_label("Project", state.project.dir_path or "none"), 
            items_maker
        )

    @staticmethod
    def ask_and_set_new_project(state: State) -> bool:
        """
        Asks user for directory and creates new project
        Returns True on success
        """
        console_message = "Enter the path to an empty directory:"
        ui_title = "Select empty directory"

        dir_path = AskUtil.ask_dir_path(
            console_message=console_message,
            ui_title=ui_title,
            initialdir=state.project.dir_path,
            mustexist=False
        )

        if not dir_path:
            return False

        old_project = state.project

        err = state.make_and_set_new_project(dir_path)
        if err:
            AskUtil.ask_error(err)
            return False
        
        if AskUtil.ask_confirm("Do you want to carry over the current project's settings?"):
            state.project.migrate_from(old_project)

        return True

    @staticmethod
    def ask_and_set_existing_project(state: State) -> bool:
        """
        Asks user for directory and if valid, sets state to existing project
        Returns True on success
        """
        s = "Enter existing project directory path:"
        s2 = "Select existing project directory"
        dir = AskUtil.ask_dir_path(s, s2, initialdir=state.project.dir_path, mustexist=True)
        if not dir:
            return False
        err = Project.is_valid_project_dir(dir)
        if err:
            AskUtil.ask_error(err)
            return False

        state.set_existing_project(dir)

        max_count = state.project.segmentation_max_words or PhraseGroup.get_max_num_words(state.project.phrase_groups)
        if max_count > DEFAULT_MAX_WORDS_PER_SEGMENT:
            message = HINT_MAX_WORDS_OVER_DEFAULT_MESSAGE
            message = message.replace("%1", str(max_count))
            message = message.replace("%2", str(DEFAULT_MAX_WORDS_PER_SEGMENT))
            hint = Hint("", "FYI", message)
            AppUtil.show_hint(hint, and_prompt=True)

        return True

# ---

def on_language(state: State, __: MenuItem) -> None:

    print_heading("Language code")
    printt(LANGUAGE_CODE_DESC)
    printt()
    if Tts.get_type() == TtsModelInfos.CHATTERBOX:
        from tts_audiobook_tool.chatterbox_model import ChatterboxModel
        printt(f"Valid values for the Chatterbox model are: {ChatterboxModel.supported_languages()}")
        printt()

    def validator(code: str) -> str:
        # Super-basic validation here; not using white-list for now
        code = code.strip()
        bad = len(code) > 5
        bad = bad or not any(char.isalpha() for char in code)
        if bad:
            return "Bad value"
        
        if ValidateUtil.is_unsupported_language_code(code):
            # Show hint as a "side effect"
            text = HINT_VALIDATION_UNSUPPORTED_LANGUAGE.text.replace("%1", str(VALIDATION_UNSUPPORTED_LANGUAGES))
            hint = replace(HINT_VALIDATION_UNSUPPORTED_LANGUAGE, text=text)
            AppUtil.show_hint_if_necessary(state.prefs, hint, and_prompt=True)
            
        return ""

    VoiceMenuShared.ask_string_and_save(
        state.project,
        f"Enter language code:",
        "language_code",
        "Language set to:",
        validator=validator
    )

LANGUAGE_CODE_DESC = "" + \
"""Language code is used by the app at various stages of the pipeline as a \"hint\" for:
- Segmentation of imported text by sentence
- TTS prompt pre-processing 
- Whisper transcription
- Text-to-speech inference (required by Chatterbox; not utilized by the other supported models)"""
