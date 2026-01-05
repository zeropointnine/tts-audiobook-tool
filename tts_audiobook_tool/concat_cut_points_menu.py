from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class ConcatCutPointsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def on_clear(_: State, __: MenuItem) -> None:
            state.project.section_dividers = []
            state.project.save()
            print_feedback("Cleared cut points")

        items = []
        items.append( MenuItem("Enter comma-delimited list", lambda _, __: ask_cut_points(state)) )
        items.append( MenuItem("Enter regular expression", lambda _, __: ask_cut_points_regex(state)) )
        if state.project.section_dividers:
            items.append( MenuItem("Clear", on_clear) )
        items.append( MenuItem("Print chapter list", lambda _, __: print_list(state)))

        MenuUtil.menu(state, ConcatCutPointsMenu.make_cut_points_label, items)

    @staticmethod
    def make_cut_points_label(state: State) -> str:
        qty = len(state.project.section_dividers)
        if state.project.section_dividers:
            label = "Chapter cut points "
            label += make_currently_string(f"{qty} cut {make_noun('point', 'points', qty)}")
        else:
            label = "Define chapter cut points "
            label += f"{COL_DIM}(optional)"
        return label



# ---

def ask_cut_points(state: State) -> None:

    num_text_groups = len(state.project.phrase_groups)

    section_dividers = state.project.section_dividers
    if section_dividers:
        print_cut_points(section_dividers, num_text_groups)

    printt("Enter the line numbers where new chapter files will begin.")
    printt(f"{COL_DIM}For example, if there are 400 lines of text and you enter \"101, 201\",")
    printt(f"{COL_DIM}three files will be created spanning lines 1-100, 101-200, and 201-400.")
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
    print_feedback(f"Cut points set:", s)

def ask_cut_points_regex(state: State) -> None:

    num_text_groups = len(state.project.phrase_groups)

    section_dividers = state.project.section_dividers
    if section_dividers:
        print_cut_points(section_dividers, num_text_groups)

    printt("Enter a regex pattern to find lines where new chapter files will begin.")
    printt(f"{COL_DIM}For example, if chapters start with \"Chapter 241: Chapter Name\", you could enter \"Chapter \\d+\",")
    printt(f"{COL_DIM}which will insert a section divider for each line matching that pattern.")
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
    print_feedback(f"Cut points set:", s)

def print_cut_points(section_dividers: list[int], num_items: int) -> None:
    section_index_strings = [str(index+1) for index in section_dividers]
    section_indices_string = ", ".join(section_index_strings)
    ranges = make_section_ranges(section_dividers, num_items)
    range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
    ranges_string = ", ".join(range_strings)
    printt(f"Current cut points: {section_indices_string} {COL_DIM}({ranges_string})")
    printt()

def print_list(state: State) -> None:

    print_heading("Chapter list:")
    printt()

    infos = ChapterInfo.make_chapter_infos(state.project)
    
    for index, info in enumerate(infos):

        s = f"{COL_DEFAULT}[{COL_ACCENT}{index+1}{COL_DEFAULT}] "
        len_prefix = len(strip_ansi_codes(s))
        s += f"Lines {info.segment_index_start + 1}-{info.segment_index_end + 1} "
        s += f"({info.num_files_exist}/{info.num_segments} generated){COL_DEFAULT}"
        if index == 0:
            s += f" {Ansi.ITALICS}{COL_DIM}(implicit)"
        printt(s)

        text = ellipsize(state.project.phrase_groups[info.segment_index_start].presentable_text, 50)
        printt(f"{' ' * len_prefix}{COL_DIM}{text}")
    printt()

    # Also print cut point indices
    one_based_indices = [info.segment_index_start + 1 for info in infos[1:]] # skip first item
    indices_string = ", ".join([str(i) for i in one_based_indices])
    printt(f"{COL_ACCENT}Cut points: {COL_DEFAULT}{indices_string}")
    printt()
