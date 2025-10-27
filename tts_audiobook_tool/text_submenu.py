from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextSubmenu:

    @staticmethod
    def replace_text_menu(state: State) -> None:

        def make_heading(_) -> str:
            s = str(len(state.project.text_segments))
            s += " line" if len(state.project.text_segments) == 1 else " lines"
            return f"Text {make_currently_string(s)}"

        items = [
            MenuItem("Replace text", on_ask_replace),
            MenuItem("View text lines", lambda _, __: print_project_text(state))
        ]
        MenuUtil.menu(state, make_heading, items)

    @staticmethod
    def set_text_menu(state: State) -> None:
        items = [
            MenuItem("Import from text file", on_set_text, data="import"),
            MenuItem("Manually enter/paste text", on_set_text, data="manual")
        ]
        MenuUtil.menu(state, "Set text", items, hint=HINT_LINE_BREAKS)

# ---

def on_ask_replace(state: State, _) -> None:

    num_files = state.project.sound_segments.num_generated()
    if num_files == 0:
        TextSubmenu.set_text_menu(state)
        return

    s = f"Replacing text will invalidate all ({num_files}) previously generated sound segment files for this project.\n"
    s += "Are you sure? "
    if AskUtil.ask_confirm(s):
        TextSubmenu.set_text_menu(state)

def on_set_text(state, item: MenuItem) -> bool:
    if item.data == "import":
        text_segments, raw_text = AppUtil.get_text_segments_from_ask_text_file()
    elif item.data == "manual":
        text_segments, raw_text = AppUtil.get_text_segments_from_ask_std_in()
    else:
        return False
    if not text_segments:
        return False
    return finish_set_text(state, text_segments, raw_text)

def finish_set_text(state: State, text_segments: list[TextSegment], raw_text: str) -> bool:

    # Print text segments
    strings = [item.text for item in text_segments]
    AppUtil.print_text_segment_text(strings)
    printt(f"{COL_DIM}... is how the text will be segmented for inference.")
    printt()

    # Confirm
    if not AskUtil.ask_confirm():
        return False

    # Delete now-outdated gens
    old_sound_segments = state.project.sound_segments.sound_segments
    for path in old_sound_segments.values():
        delete_silently(path)

    # Commit
    state.project.set_text_segments_and_save(text_segments, raw_text=raw_text)

    if not state.real_time.custom_text_segments:
        state.real_time.line_range = None

    print_feedback("Project text has been set")
    return True

def print_project_text(state: State) -> None:

    indices = state.project.sound_segments.sound_segments.keys()
    text_segments = state.project.text_segments

    printt(f"{COL_ACCENT}Text segments ({COL_DEFAULT}{len(text_segments)}{COL_ACCENT}):")
    printt()

    max_width = len(str(len(text_segments)))

    for i, text_segment in enumerate(text_segments):
        s1 = make_hotkey_string( str(i+1).rjust(max_width) )
        s2 = make_hotkey_string("y" if i in indices else "n")
        printt(f"{s1} {s2}  {text_segment.text.strip()}")
    printt()

    printt(f"{COL_DIM}Num generated audio segments: {COL_ACCENT}{len(indices)} {COL_DIM}/ {COL_ACCENT}{len(text_segments)}")
    printt()
    AskUtil.ask_enter_to_continue()
