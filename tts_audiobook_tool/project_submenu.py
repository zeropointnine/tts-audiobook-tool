from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.dir_open_util import DirOpenUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.voice_submenu_shared import VoiceSubmenuShared

class ProjectSubmenu:

    @staticmethod
    def menu(state:State) -> None:

        def on_project(_, menu_item) -> bool:
            is_new: bool = menu_item.data
            if is_new:
                did = ProjectSubmenu.ask_and_set_new_project(state)
            else:
                did = ProjectSubmenu.ask_and_set_existing_project(state)
            if did:
                print_feedback("Project directory set:", state.project.dir_path)
                if is_new:
                    AppUtil.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS, and_prompt=True)
                return True
            return False

        def make_view_label(_) -> str:
            return f"Show directory in OS UI {COL_DIM}({state.project.dir_path})"

        def on_view(_, __) -> None:
            err = DirOpenUtil.open(state.project.dir_path)
            if err:
                AskUtil.ask_error(err)

        def make_language_label(_) -> str:
            s = make_currently_string(state.project.language_code or "none")
            return f"Language code {s}"

        def on_clear_language(_, __) -> None:
            state.project.language_code = ""
            state.project.save()
            print_feedback("Language code cleared")

        # Menu
        def make_heading(_) -> str:
            s = make_currently_string(state.project.dir_path or "none")
            return f"Project {s}"

        def items_maker(_) -> list[MenuItem]:
            items = [
                MenuItem("New project", on_project, data=True),
                MenuItem("Open existing project", on_project, data=False)
            ]
            if state.project.dir_path:
                items.append(
                    MenuItem(make_view_label, on_view)
                )
            items.append(MenuItem(make_language_label, on_language))
            if state.project.language_code:
                items.append(MenuItem("Clear language code", on_clear_language))
            return items

        MenuUtil.menu(state, make_heading, items_maker)

    @staticmethod
    def ask_and_set_new_project(state: State) -> bool:
        """
        Asks user for directory and creates new project
        Returns True on success
        """
        console_message = "Enter the path to an empty directory:"
        ui_title = "Select empty directory"

        # FYI: GTK-based folder requestor dialog has no "new folder" functionality
        # but there are no good alternatives IMO, so 

        dir_path = AskUtil.ask_dir_path(
            console_message=console_message, 
            ui_title=ui_title, 
            initialdir=state.project.dir_path, 
            mustexist=False
        )

        if not dir_path:
            return False
        
        err = state.make_new_project(dir_path)
        if err:
            AskUtil.ask_error(err)
            return False
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

    def validator(s: str) -> str:
        # Super-basic validation here
        # Not using white-list for now; consider a project-language-code-to-x/y/z mapping in the future, if need be
        bad = len(s) > 5
        bad = bad or not any(char.isalpha() for char in s)
        return "Bad value" if bad else ""

    VoiceSubmenuShared.ask_string_and_save(
        state.project,
        f"Enter language code:",
        "language_code",
        "Language set to:",
        validator=validator
    )

LANGUAGE_CODE_DESC = "" + \
"""Language code is used by the app at various stages of the pipeline as a \"hint\" to improve:
- Imported text segmentation by sentence
- TTS prompt pre-processing 
- Whisper transcription (which is used to validate TTS output)
- Text-to-speech inference (required by Chatterbox; not utilized by the other supported models)"""
