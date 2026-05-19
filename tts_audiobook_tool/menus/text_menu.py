from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_support import app_display
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor, EpubImportResult
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool import ask_phrase_groups
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
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

        def on_print(_: State, __: MenuItem) -> None:
            app_display.print_project_text(
                phrase_groups=state.project.phrase_groups,
                extant_indices = set( state.project.sound_segments.sound_segments_map.keys() ),
                language_code_used=state.project.applied_language_code,
                max_words_used=state.project.applied_max_words,
                strategy_used=state.project.applied_strategy or list(SegmentationStrategy)[0]
            )
            ask.ask_enter_to_continue()

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
            state.project.set_phrase_groups_and_save(
                phrase_groups=[],
                strategy=state.project.segmentation_strategy,
                max_words=state.project.max_words,
                language_code=state.project.language_code,
                raw_text=""
            )

            if not state.real_time.custom_phrase_groups:
                state.real_time.project_text_line_range = None

            print_feedback("Project text cleared")

        items = []
        items.append(
            MenuItem("Import from text file", on_set_text, data="import"),
        )
        items.append(
            MenuItem("Import from epub file", on_set_text, data="epub"),
        )
        items.append(
            MenuItem("Import manually (input or paste text)", on_set_text, data="manual"),
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

        items.append(
            MenuItem(
                "Print current text segments", on_print,
                superlabel=" ", superlabel_no_blank_line=True
            ),
        )

        MenuUtil.menu(
            state, make_heading, items, 
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
            phrase_groups, raw_text = ask_phrase_groups.get_from_text_file(
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
            if not phrase_groups:
                print_feedback("Cancelled")
                return False

        case "epub":
            epub_path = ask_epub_path(state)
            if not epub_path:
                return False
            try:
                printt(f"{COL_DIM_ITALICS}Importing epub file... ")
                epub_import_result = EpubExtractor.import_epub(
                    epub_path=epub_path,
                    max_words=state.project.max_words,
                    segmentation_strategy=state.project.segmentation_strategy,
                    language_code=state.project.language_code,
                )
            except ImportError as e:
                printt()
                ask.ask_error(str(e))
                return False
            except Exception as e:
                printt()
                ask.ask_error(f"Error importing EPUB: {e}")
                return False

            phrase_groups = epub_import_result.phrase_groups
            raw_text = epub_import_result.raw_text
            if not phrase_groups:
                printt()
                ask.ask_enter_to_continue("No text segments.")
                return False

            # Print info/warnings
            printt()
            printt(f"{COL_ACCENT}Import info:{COL_DEFAULT}")
            for warning in epub_import_result.significant_warnings:
                printt(f"- {warning}")
            printt()

            # Print dividers info
            printt(f"{COL_ACCENT}Section markers:")
            num_dividers = len(epub_import_result.section_dividers)
            verb = "was" if num_dividers == 1 else "were"
            noun = make_noun('section marker', 'section markers', num_dividers)
            printt(f"{num_dividers} {noun} {verb} set using the epub's built-in structure.")
            printt("They can be reviewed and edited at Main > Create > Section markers")
            printt()

            # Print raw text conversion info
            printt(f"{COL_ACCENT}Text file conversion:")
            raw_text_path = os.path.join(state.project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
            raw_text_link = text_util.make_terminal_hyperlink(raw_text_path, raw_text_path, is_file=True)
            printt(f"The plain text conversion of the epub can be reviewed here:")
            printt(f"{raw_text_link}")
            printt()

            ask.ask_enter_to_continue("Press enter to review text segmentation info: ", is_replacement=True)

        case _:
            raise ValueError(f"Bad value: {item.data!r}")
    
    # Preview text segments
    app_display.print_project_text(
        phrase_groups=phrase_groups,
        extant_indices=None,
        language_code_used=state.project.language_code,
        max_words_used=state.project.max_words,
        strategy_used=state.project.segmentation_strategy
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
        state.project.set_phrase_groups_chapters_and_save(
            phrase_groups=phrase_groups,
            section_dividers=epub_import_result.section_dividers,
            strategy=state.project.segmentation_strategy,
            max_words=state.project.max_words,
            language_code=state.project.language_code,
            raw_text=raw_text
        )
    else:
        state.project.set_phrase_groups_and_save(
            phrase_groups=phrase_groups,
            strategy=state.project.segmentation_strategy,
            max_words=state.project.max_words,
            language_code=state.project.language_code,
            raw_text=raw_text
        )

    if not state.real_time.custom_phrase_groups:
        state.real_time.project_text_line_range = None

    if epub_import_result:
        print_feedback(
            "Project text has been set from EPUB:",
            f"{len(epub_import_result.chapters)} chapters, {len(epub_import_result.section_dividers)} dividers"
        )
    else:
        print_feedback("Project text has been set")
    return True

def ask_epub_path(state: State) -> str:
    if state.prefs.last_text_dir and os.path.exists(state.prefs.last_text_dir):
        initial_dir = state.prefs.last_text_dir
    else:
        initial_dir = ""
    path = ask.ask_file_path(
        "Enter EPUB file path: ",
        "Select EPUB file",
        filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")],
        initialdir=initial_dir
    )
    if not path:
        return ""
    if not os.path.exists(path):
        ask.ask_error("No such file")
        return ""
    if os.path.splitext(path)[1].lower() != ".epub":
        ask.ask_error("Must select an .epub file")
        return ""
    state.prefs.last_text_dir = str(Path(path).parent)
    return path

def on_ask_max_size(state: State, _) -> None:

    MenuUtil.print_screen_heading(state, "Max words per segment")

    printt("On text import, this is the maximum number of words to be used for a single text segment.")
    printt(f"Recommended range for current model: {COL_ACCENT}{TtsModelInfos.recommended_range_string(Tts.get_type().value)}")
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

SUBHEADING = \
"""On import, text will be segmented into sentences and phrases using the settings 
shown below. Project language code can also affect how the text is segmented.
"""
