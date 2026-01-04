from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.real_time_util import RealTimeUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *


class RealTimeMenu:

    @staticmethod
    def menu(state: State):

        def make_text_label(_) -> str:
            if state.real_time.custom_phrase_groups:
                num = len(state.real_time.custom_phrase_groups)
                value = f"custom text, {num} {make_noun('line', 'lines', num)}"
            else:
                value = "project text"
            return make_menu_label("Text source", value)

        def make_range_label(_) -> str:
            line_range = state.real_time.line_range
            if line_range:
                value = f"{line_range[0]}-{line_range[1]}"
            else:
                value = "all"
            return make_menu_label("Select line range", value)

        # Menu        
        items = [
            MenuItem("Start", lambda _, __: do_start(state)),
            MenuItem(make_text_label, lambda _, __: RealTimeMenu.text_menu(state)),
            MenuItem(make_range_label, lambda _, __: RealTimeMenu.ask_line_range(state)),
            MenuItem(
                lambda _: make_menu_label("Save output", state.project.realtime_save),
                lambda _, __: RealTimeMenu.save_menu(state)
            )
        ]
        MenuUtil.menu(state, "Real-time audio generation", items, hint=HINT_REAL_TIME)

    @staticmethod
    def ask_line_range(state: State) -> None:

        if state.real_time.custom_phrase_groups:
            text_groups = state.real_time.custom_phrase_groups
        else:
            text_groups = state.project.phrase_groups
        length = len(text_groups)

        s = "Enter line range (eg, \"5-15\"; \"50\" for 50 to end; or \"all\")"
        printt(s)
        inp = AskUtil.ask()
        if not inp:
            return
        result = ParseUtil.parse_range_string_normal(inp, length)
        if isinstance(result, str):
            AskUtil.ask_error(result)
            return

        state.real_time.line_range = result

        # Print feedback
        is_all = (result[0] == 0 and result[1] == 0) or (result[0] == 1 and result[1] == len(text_groups))
        if is_all:
            value = f"1-{len(text_groups)} (all)"
        else:
            value = f"{result[0]}-{result[1]}"
            if result[1] == len(text_groups):
                value += " (end)"
        print_feedback("Line range set:", value)

    @staticmethod
    def text_menu(state: State) -> None: # type: ignore

        # 1
        def on_project(_: State, __: MenuItem) -> bool:
            if state.real_time.custom_phrase_groups:
                state.real_time.custom_phrase_groups = []
                state.real_time.line_range = None
            print_feedback("Text source set to", "project")
            return True

        project_item = MenuItem("Use project text", on_project)

        # 2, 3
        def on_custom(_: State, item: MenuItem) -> bool:
            if item.data == "file":
                phrase_groups, __ = AppUtil.get_phrase_groups_from_ask_text_file(
                    state.project.max_words, 
                    state.project.segmentation_strategy, 
                    pysbd_language=state.project.language_code,
                    prefs=state.prefs
                )
            else:
                phrase_groups, __ = AppUtil.get_text_groups_from_ask_std_in(
                    state.project.max_words, state.project.segmentation_strategy, pysbd_language=state.project.language_code)
            if phrase_groups:
                state.real_time.custom_phrase_groups = phrase_groups
                state.real_time.line_range = None
                print_feedback("Text source set to: custom", f"{len(phrase_groups)} {make_noun('line', 'lines', len(phrase_groups))}")
            return bool(phrase_groups)

        custom_file_item = MenuItem("Custom text - from text file", on_custom, data="file")
        custom_manual_item = MenuItem("Custom text - manual input", on_custom, data="manual")

        # Menu
        items = [project_item, custom_file_item, custom_manual_item]
        MenuUtil.menu(state, "Real-time - Set text source", items, hint=HINT_REAL_TIME)

    @staticmethod
    def save_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.realtime_save = value
            state.project.save()
            print_feedback(f"Set to:", state.project.realtime_save)

        subheading = f"Saves FLAC files to {state.project.realtime_path}\n"

        MenuUtil.options_menu(
            state=state,
            heading_text="Save output to files",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.realtime_save,
            default_value=PROJECT_DEFAULT_REALTIME_SAVE,
            on_select=on_select
        )

# ---

def do_start(state: State) -> None:
    if state.real_time.custom_phrase_groups:
        text_groups = state.real_time.custom_phrase_groups
    else:
        text_groups = state.project.phrase_groups
    if not text_groups:
        print_feedback("No text segments specified")
        return

    err = Tts.check_valid_language_code(state.project)
    if err:
        print_feedback(err)
        return

    if AskUtil.is_readchar:
        b = AskUtil.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
        if not b:
            return

    AppUtil.show_pre_inference_hints(state.prefs, state.project)

    RealTimeUtil.start(
        state=state,
        phrase_groups=text_groups,
        line_range=state.real_time.line_range
    )
