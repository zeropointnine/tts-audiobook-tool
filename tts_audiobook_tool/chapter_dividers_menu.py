from tts_audiobook_tool.app_types import ChapterMode
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class ChapterDividersMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_chapter_dividers_label(_: State) -> str:
            num = len(state.project.section_dividers)
            value = f"{num} {make_noun('item', 'items', num)}"
            return make_menu_label("Chapter dividers", value)
            
        def make_mode_label(_: State) -> str:
            return make_menu_label("Mode", state.project.chapter_mode.label, list(ChapterMode)[0])
        
        def on_clear(_: State, __: MenuItem) -> None:
            state.project.section_dividers = []
            state.project.save()
            print_feedback("Cleared chapter dividers")

        items = []
        items.append( MenuItem("Enter comma-delimited list", lambda _, __: ask_cut_points(state)) )
        items.append( MenuItem("Enter regular expression", lambda _, __: ask_cut_points_regex(state)) )
        if state.project.section_dividers:
            items.append( MenuItem("Clear", on_clear) )
        items.append( MenuItem(make_mode_label, lambda _, __: mode_menu(state)) )
        items.append( MenuItem("Print chapter info", lambda _, __: print_list(state)))

        MenuUtil.menu(state, make_chapter_dividers_label, items)

# ---

def ask_cut_points(state: State) -> None:

    num_text_groups = len(state.project.phrase_groups)

    section_dividers = state.project.section_dividers
    if section_dividers:
        print_cut_points(section_dividers, num_text_groups)

    printt("Enter the line numbers where new chapters will begin.")
    printt(f"{COL_DIM}For example, if there are 400 lines of text and you enter \"101, 201\",")
    printt(f"{COL_DIM}it will create three sections spanning lines 1-100, 101-200, and 201-400.")
    printt()
    inp = AskUtil.ask()
    if not inp:
        return

    string_items = inp.split(",")
    one_indexed_items = []
    for string_item in string_items:
        try:
            index = int(string_item)
            one_indexed_items.append(index)
        except:
            AskUtil.ask_error(f"Parse error: {string_item}")
            return
    one_indexed_items = list(set(one_indexed_items))
    one_indexed_items.sort()
    for item in one_indexed_items:
        if item < 1 or item > len(state.project.phrase_groups):
            AskUtil.ask_error(f"Index out of range: {item}")
            return
    zero_indexed_items = [item - 1 for item in one_indexed_items]
    if 0 in zero_indexed_items:
        del zero_indexed_items[0]
    state.project.section_dividers = zero_indexed_items
    state.project.save()

    if not zero_indexed_items:
        s = "none"
    else:
        s = ", ".join( [str(item + 1) for item in zero_indexed_items] )
    print_feedback(f"Chapter dividers set:", s)

def ask_cut_points_regex(state: State) -> None:

    num_text_groups = len(state.project.phrase_groups)

    section_dividers = state.project.section_dividers
    if section_dividers:
        print_cut_points(section_dividers, num_text_groups)

    printt("Enter a regex pattern to define chapter dividers.")
    printt(f"{COL_DIM}For example, if the source text has chapters that start with, eg, \"Chapter 241: Chapter Name\", you could enter \"Chapter \\d+\",")
    printt(f"{COL_DIM}which will insert a chapter divider for each line matching that pattern.")
    printt()
    inp = AskUtil.ask(lower=False)
    if not inp:
        return

    zero_indexed_items = []
    pattern = re.compile(inp, flags=re.IGNORECASE)
    for index, phrase_group in enumerate(state.project.phrase_groups):
        if re.match(pattern, phrase_group.text):
            zero_indexed_items.append(index)

    # Ignore first chapter marker if it's 0 since these are dividers
    first_item_implicit = zero_indexed_items and zero_indexed_items[0] == 0
    if first_item_implicit:
        del zero_indexed_items[0]

    if not zero_indexed_items and not first_item_implicit:
        print_feedback("No matches found", is_error=True)
        return

    def print_line(i: int):
        s = f"{COL_DEFAULT}Line {i + 1}: {COL_DIM}{state.project.phrase_groups[i].presentable_text}"
        if i == 0:
            s += f" {Ansi.ITALICS}(implicit)"
        print(s)

    print_heading("Matched chapter dividers:", non_menu=True)
    printt()
    if first_item_implicit:
        print_line(0)
    for index in zero_indexed_items:
        print_line(index)
    printt()

    if not AskUtil.ask_confirm():
        print_feedback("Cancelled")
        return

    state.project.section_dividers = zero_indexed_items
    state.project.save()

    s = ", ".join( [str(item + 1) for item in zero_indexed_items] )
    print_feedback(f"Chapter dividers set:", s)

def print_cut_points(section_dividers: list[int], num_items: int) -> None:
    section_index_strings = [str(index+1) for index in section_dividers]
    section_indices_string = ", ".join(section_index_strings)
    ranges = make_chapter_ranges(section_dividers, num_items)
    range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
    ranges_string = ", ".join(range_strings)
    printt(f"Current chapter dividers: {section_indices_string} {COL_DIM}({ranges_string})")
    printt()

def print_list(state: State) -> None:

    print_heading("Chapter info:")
    printt()

    infos = ChapterInfo.make_chapter_infos(state.project)

    # First print list of chapter dividers
    one_based_indices = [info.segment_index_start + 1 for info in infos[1:]] # skip first item
    indices_string = ", ".join([str(i) for i in one_based_indices])
    printt(f"{COL_ACCENT}Chapter dividers: {COL_DEFAULT}{indices_string}")
    printt()

    is_files = (state.project.chapter_mode == ChapterMode.FILES)

    if not is_files:
        infos = infos[1:] # messy

    for index, info in enumerate(infos):

        qual = "File " if is_files else "Chapter divider "
        s = f"{COL_DEFAULT}[{COL_ACCENT}{qual}{index+1}{COL_DEFAULT}] "
        len_prefix = len(strip_ansi_codes(s))
        s += f"Lines {info.segment_index_start + 1}-{info.segment_index_end + 1} "
        s += f"({info.num_files_exist}/{info.num_segments} generated){COL_DEFAULT}"
        if is_files and index == 0:
            s += f" {Ansi.ITALICS}{COL_DIM}(implicit)"
        printt(s)

        qual = f"{COL_DIM}{Ansi.ITALICS}Starts with:{Ansi.RESET} " if is_files else ""
        text = ellipsize(state.project.phrase_groups[info.segment_index_start].presentable_text, 50)
        printt(f"{' ' * len_prefix}{qual}{COL_DIM}{text}")
    
    printt()

def mode_menu(state: State) -> None:

    def on_select(value: ChapterMode) -> None:
        state.project.chapter_mode = value
        state.project.save()
        print_feedback(f"Chapter mode set to: {value.label}")

    chapter_modes = list(ChapterMode)

    MenuUtil.options_menu(
        state=state,
        heading_text="Chapter divider behavior/mode",
        labels=[item.label for item in chapter_modes],
        values=[item for item in chapter_modes],
        current_value=state.project.chapter_mode,
        default_value=chapter_modes[0],
        on_select=on_select,
        sublabels=[item.description for item in chapter_modes]
    )
