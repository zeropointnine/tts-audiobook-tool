from dataclasses import replace
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.dir_open_util import DirOpenUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_util import ProjectUtil
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.voice_menu_shared import VoiceMenuShared

class ProjectMenu:

    @staticmethod
    def menu(state:State) -> None:

        def on_new_project(_: State, __: MenuItem) -> bool:            
            
            did = ProjectMenu.ask_and_set_new_project(state)
            if not did:
                return False

            print_feedback("Project directory set:", state.project.dir_path)
            Hint.show_hint_if_necessary(state.prefs, HINT_PROJECT_SUBDIRS, and_prompt=True)
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
                items.append(MenuItem(make_subst_label, lambda _, __: ProjectMenu.word_substitutions_menu(state)))

                items.append(MenuItem("Show directory in system file browser", on_view))
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

        # Show over-default-max-words hint
        max_count = state.project.applied_max_words 
        # TODO: Not doing this for now: ... or PhraseGroup.get_max_num_words(state.project.phrase_groups)
        if max_count > Tts.get_type().value.max_words_default:
            message = HINT_MAX_WORDS_OVER_DEFAULT_MESSAGE
            message = message.replace("%1", str(max_count))
            message = message.replace("%2", str(Tts.get_type().value.max_words_default))
            hint = Hint("", "FYI", message)
            Hint.show_hint(hint, and_prompt=True)

        return True

    @staticmethod
    def word_substitutions_menu(state: State) -> None:

        def on_enter(_, __) -> None:
            inp = AskUtil.ask(SUBSTITUTIONS_ASK_DESC, lower=False)
            if not inp:
                return 
            # Add curlies
            if not inp.startswith("{"):
                inp = "{" + inp
            if not inp.endswith("}"):
                inp = inp + "}"
            result = ProjectUtil.parse_word_substitutions_json_string(inp)
            if isinstance(result, str):
                print_feedback(result, is_error=True)
                return 
            state.project.word_substitutions = result
            state.project.save()
            print_feedback("Set word substitutions")
            return 

        def on_clear(_, __) -> None:
            state.project.word_substitutions = {}
            state.project.save()
            print_feedback("Cleared")

        def on_print(_, __) -> None:
            print_heading("Current word substitutions")
            s = str(state.project.word_substitutions)
            printt(s)
            print_feedback("", extra_line=False)
            return 
        
        def on_inspect(_, __) -> None:
            print_heading("Uncommon words")
            printt(UNCOMMON_WORDS_DESC)
            
            all_words_raw = [] 
            for group in state.project.phrase_groups:
                for phrase in group.phrases:
                    all_words_raw.extend(phrase.words)

            items = TextUtil.get_uncommon_words_en(all_words_raw)
            if not items:
                printt("None found")
            else:
                for i in range(0, min(len(items), 25)):
                    item = items[i]
                    word_str = f"{COL_DEFAULT}{item[0]}"
                    num_str = f"{COL_DIM}{str(item[1]).rjust(3)}"
                    instances_str = f"{COL_DEFAULT}{', '.join(item[2])}"
                    print(f"{num_str}  {instances_str}")
            print_feedback("", extra_line=False)

        def items_maker(_) -> list[MenuItem]:
            items = []
            # Enter items
            verb = "Replace" if state.project.word_substitutions else "Enter"
            items.append( MenuItem(f"{verb} word substitutions", on_enter) )
            # Clear items
            if state.project.word_substitutions:
                items.append(MenuItem("Clear", on_clear))
            # Print items
            if state.project.word_substitutions:
                label = make_menu_label("Print current list", f"{len(state.project.word_substitutions)} items")
                items.append( MenuItem(label, on_print) )
            # Print uncommon words
            if state.project.language_code == "en" and state.project.phrase_groups:
                items.append(MenuItem("Inspect project text for uncommon words", on_inspect))
            return items

        MenuUtil.menu(
            state, 
            heading=make_subst_label,
            items=items_maker,
            subheading=SUBSTITUTIONS_DESC
        )

# ---

def make_subst_label(state: State) -> str:
    num_subst = len(state.project.word_substitutions)
    value = f"{num_subst} {make_noun('item', 'items', num_subst)}" if num_subst > 0 else "none"
    label = f"Word substitutions {make_currently_string(value)}"
    return label

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
            Hint.show_hint(hint, and_prompt=True)

        return ""

    VoiceMenuShared.ask_string_and_save(
        state.project,
        f"Enter language code:",
        "language_code",
        "Language set to:",
        validator=validator
    )

    # Special case
    if state.project.language_code != "en" and state.project.strictness != Strictness.LOW:
        if not ValidateUtil.is_unsupported_language_code(state.project.language_code):
            state.project.strictness = Strictness.LOW
            state.project.save()
            printt(FORCED_STRICTNESS_LOW_DESC)
            AskUtil.ask_enter_to_continue()


LANGUAGE_CODE_DESC = "" + \
"""Language code is used by the app at various stages of the pipeline as a \"hint\" for:
- Semantic segmentation of imported text
- Prompt pre-processing 
- Whisper transcription
- TTS inference (required by Chatterbox)"""

SUBSTITUTIONS_DESC = \
f"""List of words to be replaced in the TTS prompt at inference-time.
Useful for helping the model pronounce proper names, neologisms, etc. 
more accurately. {COL_DIM}(Requires some trial and error){COL_DEFAULT}
"""

SUBSTITUTIONS_ASK_DESC = \
f"""Enter substitutions list. Use this format: 
{COL_DIM}{Ansi.ITALICS}{{"Ariekei": "AriaKay", "kilohour": "kilo hour"}}
 
"""

UNCOMMON_WORDS_DESC = \
f"""Most frequent words in the project text not found 
in the app's English \"common words\" dictionary.
"""

FORCED_STRICTNESS_LOW_DESC = \
f"""{COL_ACCENT}Note: {COL_DEFAULT}Because the language code is not en, the setting \"Transcript validation strictness\" 
has been automatically set to \"Low\"
"""