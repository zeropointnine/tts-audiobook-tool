from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.app_support import app_hint_util
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool import readiness
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual.real_time_playback_app import (
    run_real_time_playback_modal,
)
from tts_audiobook_tool.util import *

class RealTimePlaybackMenu:

    @staticmethod
    def menu(state: State):

        def make_start_label(_: State) -> str:
            label = "Start"
            err = readiness.get_generate_blocker_text(state, verbose=False)
            if err:
                return make_menu_label(label, err, value_prefix="", color_code=COL_ERROR)
            else:
                return label

        def make_range_label(_) -> str:
            line_range = get_active_line_range(state)
            if line_range:
                value = f"{line_range[0]}-{line_range[1]}"
            else:
                value = "all"
            return make_menu_label("Line range", value)

        # Menu
        items = [
            MenuItem(make_start_label, lambda _, __: do_start(state)),
            MenuItem(
                make_range_label, lambda _, __: RealTimePlaybackMenu.ask_line_range(state),
                superlabel="Options"
            ),
            MenuItem(
                lambda _: make_menu_label("Save output", state.project.realtime_save),
                lambda _, __: RealTimePlaybackMenu.save_menu(state)
            )
        ]
        MenuUtil.menu(
            state,
            "Realtime audiobook playback",
            items,
            subheading=REAL_TIME_SUBHEADING,
            breadcrumb="Realtime playback"
        )

    @staticmethod
    def ask_line_range(state: State) -> None:

        phrase_groups = state.project.phrase_groups
        length = len(phrase_groups)

        s = f"Enter line range {COL_DIM}(eg, \"5-15\"; \"50\" for 50 to end; or \"all\")"
        printt(s)
        line_range = get_active_line_range(state)
        prefill = f"{line_range[0]}-{line_range[1]}" if line_range else "all"
        inp = ask.ask_input(prefill=prefill)
        if not inp:
            return
        result = RangeStringUtil.parse_range_string_normal(inp, length)
        if isinstance(result, str):
            ask.ask_error(result)
            return

        if result == (0, 0):
            result = None

        if result != state.project.realtime_line_range:
            state.project.realtime_line_range = result
            state.project.save()

        # Print feedback
        is_all = result is None or (result[0] == 1 and result[1] == len(phrase_groups))
        if is_all:
            value = f"1-{len(phrase_groups)} (all)"
        else:
            assert result is not None
            value = f"{result[0]}-{result[1]}"
            if result[1] == len(phrase_groups):
                value += " (end)"
        print_feedback("Line range set:", value)

    @staticmethod
    def save_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.realtime_save = value
            state.project.save()
            print_feedback(f"Set to:", state.project.realtime_save)

        if os.path.exists(state.project.realtime_path):
            path_text = text_util.make_terminal_hyperlink(state.project.realtime_path, is_file=True)
        else:
            path_text = state.project.realtime_path
        subheading = (
            f"Saves sound segment FLAC files generated while using realtime playback to\n"
            f"{path_text}\n"
        )

        MenuUtil.options_menu(
            state=state,
            heading_text="Save output to files",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.realtime_save,
            default_value=PROJECT_DEFAULT_REALTIME_SAVE,
            on_select=on_select,
            breadcrumb="Save output",
        )

# ---

def get_active_line_range(state: State) -> tuple[int, int] | None:
    line_range = state.project.realtime_line_range
    if line_range == (0, 0):
        return None
    return line_range

def do_start(state: State) -> None:

    text_groups = state.project.phrase_groups
    line_range = get_active_line_range(state)

    if not text_groups:
        print_feedback("No text segments specified")
        return

    # Check model and other app blockers
    err = readiness.get_generate_blocker_text(state, verbose=True)
    if err:
        ask.ask_error(err)
        return

    # Show pre-inference hint/warning if necessary
    can_continue = app_hint_util.show_pre_inference_hints(state.prefs, state.project)
    if not can_continue:
        return

    # Confirm and start proper
    if ask.can_hotkey:
        b = ask.ask_confirm(f"Press {make_hotkey_string('Y')} to start: ")
        if not b:
            return

    run_real_time_playback_modal(
        state=state,
        phrase_groups=text_groups,
        line_range=line_range,
    )

REAL_TIME_SUBHEADING = (
    'Uses the same quality checks as the normal audiobook creation workflow,\n'
    'except for loudness normalization and generative upsampling.\n\n'
    'Uninterrupted playback requires inference to be faster-than-realtime.\n'
    'Validation/retry activates when there is 60 seconds of buffered audio.\n'
)
