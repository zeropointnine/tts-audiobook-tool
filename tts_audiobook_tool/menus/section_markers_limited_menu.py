from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.menu_shared import make_output_files_subheading
from tts_audiobook_tool.menus.section_markers_menu import SectionMarkersMenu
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class SectionMarkersLimitedMenu:
    """
    Menu for adjusting Project.markers for split mode only.
    Is essentially a subset of the functionality of SectionMarkersMenu.
    
    Is shown only when book text source is epub.
    The rationale for this is that epubs already have structured sections, 
    so no need for the extra functionality provided for (now deprioritized) flat text sources.
    """

    @staticmethod
    def menu(state: State) -> None:

        def on_clear(_: State, __: MenuItem) -> None:
            state.project.markers = []
            state.project.save()
            print_feedback("Cleared file split points")

        def make_items(_: State) -> list[MenuItem]:
            items = []

            items.append( 
                MenuItem(
                    "Enter line numbers", 
                    lambda _, __: SectionMarkersMenu.ask_section_markers(state, "file split points:")
                ) 
            )
            
            if state.project.markers:
                items.append( MenuItem("Clear", on_clear) )
            
            if state.project.markers:

                num = len(state.project.markers)
                value = f"{num} {make_noun('item', 'items', num)}"
                label = f"Print list {COL_DIM}({value})"

                items.append( 
                    MenuItem(
                        label, lambda _, __: print_list(state), 
                        superlabel=" ", superlabel_no_blank_line=True
                    ),
                )
            return items

        MenuUtil.menu(
            state, "File split points", make_items, subheading=make_output_files_subheading
        )

# ---

def print_list(state: State) -> None:

    MenuUtil.print_screen_heading(state, "Print list")

    for index in state.project.markers:
        text = ellipsize(state.project.phrase_groups[index].presentable_text, 60)
        s = f"Line {index+1}: {COL_DIM_ITALICS}{text}"
        printt(s)

    if state.prefs.menu_clears_screen:
        print()
        ask.ask_enter_to_continue()
    
