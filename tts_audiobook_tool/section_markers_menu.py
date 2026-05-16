from tts_audiobook_tool.app_types import SectionMarkerMode
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class SectionMarkersMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_section_markers_label(_: State) -> str:
            num = len(state.project.section_dividers)
            value = f"{num} {make_noun('item', 'items', num)}"
            return make_menu_label("Section markers", value)
            
        def make_mode_label(_: State) -> str:
            return make_menu_label("Mode", state.project.chapter_mode.label, list(SectionMarkerMode)[0])
        
        def on_clear(_: State, __: MenuItem) -> None:
            state.project.section_dividers = []
            state.project.save()
            print_feedback("Cleared section markers")

        items = []

        items.append( MenuItem("Enter comma-delimited list", lambda _, __: ask_section_markers(state)) )
        
        items.append( MenuItem("Enter regular expression", lambda _, __: ask_section_markers_regex(state)) )
        
        if state.project.section_dividers:
            items.append( MenuItem("Clear", on_clear) )
        
        items.append( MenuItem(make_mode_label, lambda _, __: mode_menu(state)) )
        
        if state.project.section_dividers:
            items.append( 
                MenuItem(
                    "Print section markers", lambda _, __: print_list(state), 
                    superlabel=" ", superlabel_no_blank_line=True
                ),
            )

        MenuUtil.menu(
            state, make_section_markers_label, items, subheading=SUBLABEL,
        )

# ---

def ask_section_markers(state: State) -> None:

    num_text_groups = len(state.project.phrase_groups)

    section_markers = state.project.section_dividers
    if section_markers:
        print_section_markers(section_markers, num_text_groups)

    printt("Enter line numbers:")
    printt(f"{COL_DIM}(Eg: 48, 101, 545)")
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
            print_feedback(f"Index out of range: {item}", is_error=True)
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
    print_feedback(f"Section markers set:", s)

def ask_section_markers_regex(state: State) -> None:

    num_text_groups = len(state.project.phrase_groups)

    section_dividers = state.project.section_dividers
    if section_dividers:
        print_section_markers(section_dividers, num_text_groups)

    printt("Enter a regex pattern to define section markers:")
    printt(f"{COL_DIM}For example, if the source text has chapters that start with, eg, \"Chapter 241: Chapter Name\", you could enter \"Chapter \\d+\",")
    printt(f"{COL_DIM}which will insert a section marker for each line matching that pattern.")
    printt()
    inp = AskUtil.ask(lower=False)
    if not inp:
        return

    zero_indexed_items = []
    pattern = re.compile(inp, flags=re.IGNORECASE)
    for index, phrase_group in enumerate(state.project.phrase_groups):
        if re.match(pattern, phrase_group.text):
            zero_indexed_items.append(index)

    # Ignore first section marker if it's 0 since these are dividers
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

    MenuUtil.print_heading(None, "Matched section markers:", non_menu=True)
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
    print_feedback(f"Section markers set:", s)

def print_section_markers(section_dividers: list[int], num_items: int) -> None:
    section_index_strings = [str(index+1) for index in section_dividers]
    section_indices_string = ", ".join(section_index_strings)
    
    if False:
        ranges = make_file_line_ranges(section_dividers, num_items)
        range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
        ranges_string = ", ".join(range_strings)
        printt(f"Current section markers: {section_indices_string} {COL_DIM}({ranges_string})")
    else:
        printt(f"Current section markers: {COL_DIM}{section_indices_string}")
    
    printt()

def print_list(state: State) -> None:

    MenuUtil.print_screen_heading(state, "Print section markers")

    for index in state.project.section_dividers:
        text = ellipsize(state.project.phrase_groups[index].presentable_text, 60)
        s = f"Line {index+1}: {COL_DIM_ITALICS}{text}"
        printt(s)

    if state.prefs.menu_clears_screen:
        print()
        AskUtil.ask_enter_to_continue()
    
def mode_menu(state: State) -> None:

    def on_select(value: SectionMarkerMode) -> None:
        state.project.chapter_mode = value
        state.project.save()
        print_feedback(f"Section marker mode set to: {value.label}")

    chapter_modes = list(SectionMarkerMode)

    MenuUtil.options_menu(
        state=state,
        heading_text="Section marker mode",
        labels=[item.label for item in chapter_modes],
        values=[item for item in chapter_modes],
        current_value=state.project.chapter_mode,
        default_value=chapter_modes[0],
        on_select=on_select,
        sublabels=[item.description for item in chapter_modes]
    )

SUBLABEL = \
"""Section markers are line numbers where new sections begin. 
Depending on the selected mode, they are used to split the audiobook into 
separate files or to add chapters/bookmarks to a single audiobook file.
"""