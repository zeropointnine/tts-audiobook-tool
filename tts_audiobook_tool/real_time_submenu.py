from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.real_time_util import RealTimeUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class RealTimeSubmenu:

    @staticmethod
    def menu(state: State):

        # 1
        def on_start(_, __) -> None:
            if state.real_time.custom_text_segments:
                text_segments = state.real_time.custom_text_segments
            else:
                text_segments = state.project.text_segments
            if not text_segments:
                print_feedback("No text segments specified")
                return
            if AskUtil.is_readchar:
                b = AskUtil.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
                if not b:
                    return
            RealTimeUtil.start(
                state=state,
                text_segments=text_segments,
                line_range=state.real_time.line_range
            )

        start_item = MenuItem("Start", on_start)

        # 2
        def make_text_label(_) -> str:
            if state.real_time.custom_text_segments:
                value = f"custom text, {len(state.real_time.custom_text_segments)} lines"
            else:
                value = "project text"
            current = make_currently_string(value)
            return f"Text source {current}"

        text_item = MenuItem(make_text_label, lambda _, __: RealTimeSubmenu.text_menu(state))

        # 3
        def make_range_label(_) -> str:
            line_range = state.real_time.line_range
            if line_range:
                value = f"{line_range[0]}-{line_range[1]}"
            else:
                value = "all"
            return f"Select line range {make_currently_string(value)}"

        range_item = MenuItem(make_range_label, lambda _, __: RealTimeSubmenu.ask_line_range(state))

        # Menu
        items = [start_item, text_item, range_item]
        MenuUtil.menu(state, "Real-time playback", items, hint=HINT_REAL_TIME)


    @staticmethod
    def ask_line_range(state: State) -> None:

        if state.real_time.custom_text_segments:
            text_segments = state.real_time.custom_text_segments
        else:
            text_segments = state.project.text_segments
        length = len(text_segments)

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
        is_all = (result[0] == 0 and result[1] == 0) or (result[0] == 1 and result[1] == len(text_segments))
        if is_all:
            value = f"1-{len(text_segments)} (all)"
        else:
            value = f"{result[0]}-{result[1]}"
            if result[1] == len(text_segments):
                value += " (end)"
        print_feedback("Line range set:", value)


    @staticmethod
    def text_menu(state: State) -> None: # type: ignore

        # 1
        def on_project(_, __) -> None:
            if state.real_time.custom_text_segments:
                state.real_time.custom_text_segments = []
                state.real_time.line_range = None
            print_feedback("Text source set to", "project")

        project_item = MenuItem("Use project text", on_project)

        # 2, 3
        def on_custom(_, item: MenuItem) -> None:
            if item.data == "file":
                text_segments, _ = AppUtil.get_text_segments_from_ask_text_file()
            else:
                text_segments, _ = AppUtil.get_text_segments_from_ask_std_in()
            if text_segments:
                state.real_time.custom_text_segments = text_segments
                state.real_time.line_range = None
                print_feedback("Text source set to: custom", f"{len(text_segments)} lines")

        custom_file_item = MenuItem("Custom text - from text file", on_custom, data="file")
        custom_manual_item = MenuItem("Custom text - manual input", on_custom, data="manual")

        # Menu
        items = [project_item, custom_file_item, custom_manual_item]
        MenuUtil.menu(state, "Real-time - Set text source", items, hint=HINT_REAL_TIME)
