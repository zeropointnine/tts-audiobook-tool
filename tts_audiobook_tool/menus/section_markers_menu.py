from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types import SectionMarkerMode
from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.menu_shared import make_output_files_subheading
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    EditorSaveFailed,
    run_content_textual_app,
)
from tts_audiobook_tool.textual.section_markers_editor import SectionMarkersEditor
from tts_audiobook_tool.util import *

class SectionMarkersMenu:
    """
    Menu for editing Project.markers.

    App UI nomenclature is "split points" when project text has multiple sections
    (ie, derived from epub), else "section markers".
    """

    @staticmethod
    def menu(state: State) -> None:

        is_limited = state.project.has_multiple_book_sections()
           
        def make_mode_label(_: State) -> str:
            return make_menu_label("Mode", state.project.chapter_mode.label, list(SectionMarkerMode)[0])
        
        def make_items(_: State) -> list[MenuItem]:

            items = []

            label = f"Edit {app_text.get_section_marker_label(state.project, is_title_case=False)}"
            num_markers = len(state.project.markers)
            item_noun = make_noun("item", "items", num_markers)
            currently = make_currently_string(f"{num_markers} {item_noun}")
            label = f"{label} {currently}"

            items.append( 
                MenuItem(
                    label,
                    lambda _, __: SectionMarkersMenu.edit_section_markers(state)
                ) 
            )

            if not is_limited:
                items.append( MenuItem(make_mode_label, lambda _, __: mode_menu(state)) )
            
            return items

        def make_subheading(state: State) -> str:
            if not is_limited:
                return SUBLABEL
            return make_output_files_subheading(state) + LIMITED_SUBLABEL

        label = app_text.get_section_marker_label(state.project)
        MenuUtil.menu(
            state, label, make_items,
            subheading=make_subheading
        )

    @staticmethod
    def edit_section_markers(state: State) -> None:
        """Run the section-markers editor and report launch failures or saves."""
        run_result = run_content_textual_app(SectionMarkersEditor(state.project))
        if not isinstance(run_result, ContentAppCompleted):
            ask.ask_error(run_result.message)
            return
        if isinstance(run_result.result, EditorSaveFailed):
            ask.ask_error(run_result.result.error)

def print_markers(markers: set[int], label: str) -> None:
    
    section_index_strings = [str(index+1) for index in sorted(markers)]
    section_indices_string = ", ".join(section_index_strings)
    printt(f"Current {label}: {COL_DIM}{section_indices_string}")    
    printt()

def print_section_markers(state: State) -> None:

    MenuUtil.print_screen_heading(state, "Print")

    if not state.project.markers:
        printt("None")
        printt()
    else:
        for index in sorted(state.project.markers):
            text = ellipsize(state.project.phrase_groups[index].presentable_text, 60)
            s = f"Line {index+1}: {COL_DIM_ITALICS}{text}"
            printt(s)

        printt()
        s = "Items: " + ", ".join(
            str(index + 1) for index in sorted(state.project.markers)
        )
        printt(s)

    print()
    ask.ask_enter_to_continue()
    
def mode_menu(state: State) -> None:

    def on_select(value: SectionMarkerMode) -> None:
        if value == SectionMarkerMode.BOOKMARKS and not state.project.can_use_bookmark_section_markers():
            ask.ask_error("Section marker mode 'Adds metadata' is unavailable for multi-section books")
            return
        state.project.chapter_mode = value
        state.project.normalize_chapter_mode()
        state.project.save()
        print_feedback(f"Section marker mode set to: {state.project.chapter_mode.label}")

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
"""Section markers are line numbers which define where new sections begin. 
Depending on the selected mode, this is used to either split the audiobook 
into separate files or to add M4B chapters and web player bookmarks to a
single audiobook file.
"""

LIMITED_SUBLABEL = \
"""File split points are line numbers that define where new audio 
files will be created.
"""
