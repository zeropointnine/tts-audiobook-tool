from tts_audiobook_tool.app_types import BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_support import app_display
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor, EpubImportResult
from tts_audiobook_tool.menus.epub_menu_util import EpubMenuUtil
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool import ask_phrase_groups
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_heading(_) -> str:
            if state.project.phrase_groups:
                if state.prefs.menu_clears_screen:
                    currently = ""
                else:
                    s = str(len(state.project.phrase_groups))
                    s += " line" if len(state.project.phrase_groups) == 1 else " lines"
                    currently = make_currently_string(s)
            else:
                currently = ""
            return f"Text {currently}"

        def make_max_size_label(_) -> str:
            value = make_parameter_value_string(
                state.project.max_words, MAX_WORDS_PER_SEGMENT_DEFAULT
            )
            return make_menu_label("Text segmentation max words per segment", value)

        def on_clear(_: State, __: MenuItem) -> None:
            num_files = state.project.sound_segments.num_generated()
            if num_files > 0:
                s = f"Clearing project text will cause all {num_files} previously generated sound segment files to be deleted.\n"
                s += "Are you sure? "
            else:
                s = "Clear project text? "
            if not ask.ask_confirm(s):
                return

            state.project.sound_segments.delete_all()
            ProjectTextIOUtil.set_phrase_groups_and_save(
                state.project,
                phrase_groups=[],
                strategy=state.project.segmentation_strategy,
                max_words=state.project.max_words,
                language_code=state.project.language_code,
                raw_text="",
                text_source_kind="manual",
            )

            if not state.real_time.custom_phrase_groups:
                state.real_time.project_text_line_range = None

            print_feedback("Project text cleared")

        def make_items(_: State) -> list[MenuItem]:

            items = []
            items.append(
                MenuItem("Import from EPUB file", on_set_text, data="epub"),
            )
            items.append(
                MenuItem("Import from plain text file", on_set_text, data="import"),
            )
            items.append(
                MenuItem("Import plain text manually (input or paste text)", on_set_text, data="manual"),
            )
            if state.project.phrase_groups:
                items.append(MenuItem("Clear text", on_clear))

            items.append(
                MenuItem(
                    make_max_size_label, on_ask_max_size, 
                    superlabel="Text import settings"
                )
            )
            items.append(
                MenuItem(
                    lambda _: make_menu_label("Text segmentation strategy", state.project.segmentation_strategy.label.lower()),
                    lambda _, __: TextMenu.strategy_menu(state)   
                )
            )

            if state.project.phrase_groups:
                
                num_sections = len(state.project.book.sections)
                if num_sections > 2:
                    items.append(
                        MenuItem(
                            f"Print sections {COL_DIM}({num_sections})", on_print_sections, 
                            superlabel=" ", superlabel_no_blank_line=True
                        )
                    )
                
                superlabel = "" if num_sections > 2 else " "
                superlabel_nbl = False if num_sections > 2 else True

                items.append(                    
                    MenuItem(
                        f"Print text segments {COL_DIM}({len(state.project.phrase_groups)})", on_print_segments,
                        superlabel=superlabel, superlabel_no_blank_line=superlabel_nbl
                    )
                )

                # items.append(
                #     MenuItem(
                #         "Print text", lambda _, __: TextMenu.print_menu(state),
                #         superlabel=" ", superlabel_no_blank_line=True
                #     ),
                # )       

            return items

        MenuUtil.menu(
            state, make_heading, make_items, 
            subheading=SUBHEADING, hint=HINT_LINE_BREAKS, breadcrumb="Text"
        )

    @staticmethod
    def strategy_menu(state: State) -> None:

        def on_select(value: SegmentationStrategy) -> None:
            state.project.segmentation_strategy = value
            state.project.save()
            print_feedback("Text segmentation strategy set to:", state.project.segmentation_strategy.label)

        MenuUtil.options_menu(
            state=state,
            heading_text="Text segmentation strategy",
            labels=[item.label for item in list(SegmentationStrategy)],
            values=[item for item in list(SegmentationStrategy)],
            sublabels=[item.description for item in list(SegmentationStrategy)],
            current_value=state.project.segmentation_strategy,
            default_value=list(SegmentationStrategy)[0],
            on_select=on_select,
            breadcrumb="Text segmentation strategy",
        )

# ---

def on_set_text(state: State, item: MenuItem) -> bool:

    num_files = state.project.sound_segments.num_generated()
    if num_files > 0:
        # First confirm
        s = f"Replacing project text will cause all {num_files} previously generated sound segment files to be deleted.\n"
        s += "Are you sure? "
        if not ask.ask_confirm(s):
            return False

    epub_path = ""
    epub_import_result: EpubImportResult | None = None

    match item.data:

        case "import":
            phrase_groups, raw_text, title = ask_phrase_groups.get_from_text_file(
                state.project.max_words,
                state.project.segmentation_strategy,
                pysbd_language=state.project.language_code,
                prefs=state.prefs
            )
            if not phrase_groups:
                return False

        case "manual":
            phrase_groups, raw_text = ask_phrase_groups.get_from_std_in(
                state.project.max_words, state.project.segmentation_strategy, pysbd_language=state.project.language_code
            )
            title = ""
            if not phrase_groups:
                print_feedback("Cancelled")
                return False

        case "epub":
            epub_path = EpubMenuUtil.ask_epub_path(state.prefs)
            if not epub_path:
                return False
            epub_import_result = EpubMenuUtil.import_epub(
                epub_path=epub_path,
                max_words=state.project.max_words,
                segmentation_strategy=state.project.segmentation_strategy,
                language_code=state.project.language_code,
            )
            if epub_import_result is None:
                return False

            phrase_groups = epub_import_result.phrase_groups
            raw_text = epub_import_result.raw_text
            title = epub_import_result.book_title
            if not phrase_groups:
                printt()
                ask.ask_enter_to_continue("No text segments.")
                return False

            # Print info/warnings
            EpubMenuUtil.print_import_info(epub_import_result)

            ask.ask_enter_to_continue("Press enter to review text segmentation info: ", is_replacement=True)

        case _:
            raise ValueError(f"Bad value: {item.data!r}")
    
    # Preview text segments
    app_display.print_book_text_lines(
        state,
        phrase_groups=phrase_groups,
        extant_indices=None,
        segmentation_settings=BookSegmentationSettings(
            language_code=state.project.language_code,
            max_words_per_segment=state.project.max_words,
            strategy=state.project.segmentation_strategy,
        ),
    )

    # Confirm
    if not ask.ask_confirm():
        print_feedback("Cancelled")
        return False

    # Delete now-outdated gens
    state.project.sound_segments.delete_all()

    # Commit
    if epub_import_result:
        err = EpubExtractor.copy_epub_to_project(epub_path, state.project.dir_path)
        if err:
            ask.ask_error(err)
            return False
        ProjectTextIOUtil.set_phrase_groups_chapters_and_save(
            state.project,
            phrase_groups=phrase_groups,
            section_start_indices=epub_import_result.section_start_indices,
            strategy=state.project.segmentation_strategy,
            max_words=state.project.max_words,
            language_code=state.project.language_code,
            raw_text=raw_text,
            title=title,
            section_titles=[chapter.title for chapter in epub_import_result.chapters],
        )

    else:
        text_source_kind = "manual" if item.data == "manual" else "plain_text"
        ProjectTextIOUtil.set_phrase_groups_and_save(
            state.project,
            phrase_groups=phrase_groups,
            strategy=state.project.segmentation_strategy,
            max_words=state.project.max_words,
            language_code=state.project.language_code,
            raw_text=raw_text,
            title=title,
            text_source_kind=text_source_kind,
        )

    if not state.real_time.custom_phrase_groups:
        state.real_time.project_text_line_range = None

    if epub_import_result:
        printt("Project text has been set")
        printt()
        
        raw_text_path = os.path.join(state.project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        raw_text_link = text_util.make_terminal_hyperlink(raw_text_path, raw_text_path, is_file=True)
        printt(f"{COL_ACCENT}A plain-text conversion{COL_DEFAULT} of the EPUB file was also saved here:")
        printt(f"{raw_text_link}")
        printt()
        ask.ask_enter_to_continue()
    else:
        print_feedback("Project text has been set")

    return True

def on_ask_max_size(state: State, _) -> None:

    MenuUtil.print_screen_heading(state, "Max words per segment")

    printt("On text import, this is the maximum number of words to be used for a single text segment.")
    printt(f"Recommended range for current model: {COL_ACCENT}{TtsModelType.recommended_range_string(Tts.get_type().value)}")
    printt()

    ask.ask_number(
        state.project,
        attr="max_words",
        prompt="Enter max words per segment:",
        min_value=MAX_WORDS_PER_SEGMENT_MIN, max_value=MAX_WORDS_PER_SEGMENT_MAX,
        default_value=MAX_WORDS_PER_SEGMENT_DEFAULT,
        success_prefix="Max words per segment set to:",
        is_int=True
    )

def on_print_sections(state: State, __: MenuItem) -> None:
    app_display.print_book_sections(state)
    ask.ask_enter_to_continue()

def on_print_segments(state: State, __: MenuItem) -> None:
    app_display.print_book_text_lines(
        state,
        phrase_groups=state.project.phrase_groups,
        extant_indices = set( state.project.sound_segments.sound_segments_map.keys() ),
        segmentation_settings=ProjectBookUtil.get_book_segmentation_settings(state.project),
    )
    ask.ask_enter_to_continue()

SUBHEADING = \
"""On import, text will be segmented into sentences and phrases using the settings 
shown below. Project language code can also affect how the text is segmented.
"""
