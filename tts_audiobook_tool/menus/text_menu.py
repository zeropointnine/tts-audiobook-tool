from tts_audiobook_tool.app_support import app_text, hints
from tts_audiobook_tool.app_types import SegmentationStrategy, VoiceSelectMode
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.text_ops import language_util
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor, EpubImportResult
from tts_audiobook_tool.text_ops.dialog_segmenter import (
    DIALOG_VOICE_INDEX,
)
from tts_audiobook_tool.text_ops.whitelist import Whitelist
from tts_audiobook_tool.menus.epub_menu_util import EpubMenuUtil
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool import ask_phrase_groups
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    EditorSaveFailed,
    EditorSaved,
    run_content_textual_app,
)
from tts_audiobook_tool.textual.text_editor import TextEditor
from tts_audiobook_tool.textual.tts_preview_app import run_tts_preview_app
from tts_audiobook_tool.textual.word_substitutions_app import (
    WordSubstitutionPreviewRequested,
    WordSubstitutionsApp,
    make_word_substitution_preview_prompt,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_heading(_) -> str:
            return "Text"

        def make_segmentation_settings_label(_) -> str:
            value = make_parameter_value_string(
                state.project.max_words, MAX_WORDS_PER_SEGMENT_DEFAULT
            )
            value = f"max words {value}; {state.project.segmentation_strategy.label.lower()}"
            if state.project.dialog_segmentation:
                value += "; dialog"
            return make_menu_label("Segmentation settings", value)

        def make_items(_: State) -> list[MenuItem]:

            items = []
            items.append(
                MenuItem("Import from EPUB file", on_select_import, data="epub"),
            )
            items.append(
                MenuItem("Import from text file", on_select_import, data="text"),
            )
            items.append(
                MenuItem("Manually enter text", on_select_import, data="manual"),
            )
            items.append(
                MenuItem(
                    make_segmentation_settings_label,
                    lambda _, __: TextMenu.segmentation_settings_menu(state),
                    superlabel=" ", superlabel_no_blank_line=True
                )
            )

            if state.project.phrase_groups:

                items.append(
                    MenuItem(
                        make_subst_label, lambda _, __: TextMenu.word_substitutions_menu(state),
                        superlabel=" ", superlabel_no_blank_line=True
                    )
                )

                items.append(
                    MenuItem(
                        "View/edit text lines", lambda _, __: TextMenu.edit_text(state),
                        superlabel=" ", superlabel_no_blank_line=True
                    ),
                )

            return items

        MenuUtil.menu(
            state, make_heading, make_items,
            hint=HINT_LINE_BREAKS, breadcrumb="Text"
        )

    @staticmethod
    def edit_text(state: State) -> None:
        """Run one text editor and present its result in the surrounding menu."""
        run_result = run_content_textual_app(TextEditor(state.project))
        if not isinstance(run_result, ContentAppCompleted):
            ask.ask_error(run_result.message)
            return
        if isinstance(run_result.result, EditorSaveFailed):
            ask.ask_error(run_result.result.error)
        elif isinstance(run_result.result, EditorSaved):
            ... # No need to print feedback here

    @staticmethod
    def segmentation_settings_menu(state: State) -> None:

        def make_max_words_label(_state: State) -> str:
            value = make_parameter_value_string(
                state.project.max_words, MAX_WORDS_PER_SEGMENT_DEFAULT
            )
            return make_menu_label("Max words per segment", value)

        def make_dialog_segmentation_label(_state: State) -> str:
            if not state.project.dialog_segmentation:
                return f"Dialog segmentation {COL_DIM}(optional)"
            return make_menu_label(
                "Dialog segmentation", state.project.dialog_segmentation
            )

        def items_maker(_) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(make_max_words_label, on_ask_max_size)
            )
            items.append(
                MenuItem(
                    lambda _state: make_menu_label(
                        "Segmentation strategy",
                        state.project.segmentation_strategy.label.lower()
                    ),
                    lambda _, __: TextMenu.strategy_menu(state)
                )
            )
            items.append(
                MenuItem(
                    make_dialog_segmentation_label,
                    lambda _, __: TextMenu.dialog_segmentation_menu(state)
                )
            )
            return items

        MenuUtil.menu(
            state,
            heading="Segmentation settings",
            items=items_maker,
            subheading=SEG_SUBHEADING,
            breadcrumb="Segmentation settings",
        )

    @staticmethod
    def strategy_menu(state: State) -> None:

        def on_select(value: SegmentationStrategy) -> None:
            state.project.segmentation_strategy = value
            state.project.save()
            print_feedback("Segmentation strategy set to:", state.project.segmentation_strategy.label)

        MenuUtil.options_menu(
            state=state,
            heading_text="Segmentation strategy",
            labels=[item.label for item in list(SegmentationStrategy)],
            values=[item for item in list(SegmentationStrategy)],
            sublabels=[item.description for item in list(SegmentationStrategy)],
            current_value=state.project.segmentation_strategy,
            default_value=list(SegmentationStrategy)[0],
            on_select=on_select,
            breadcrumb="Segmentation strategy",
        )

    @staticmethod
    def dialog_segmentation_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.dialog_segmentation = value
            state.project.save()
            print_feedback("Dialog segmentation set to:", str(value))

        MenuUtil.options_menu(
            state=state,
            heading_text="Dialog segmentation",
            subheading=DIALOG_SEGMENTATION_DESC,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.dialog_segmentation,
            default_value=False,
            on_select=on_select,
            breadcrumb="Dialog segmentation",
        )

    @staticmethod
    def edit_word_substitutions(state: State) -> bool:
        """Run the editor and its non-persistent pronunciation preview loop."""
        staged: dict[str, str] | None = None
        restore_original: str | None = None
        preview_sound = None

        while True:
            run_result = run_content_textual_app(
                WordSubstitutionsApp(
                    state,
                    staged=staged,
                    restore_original=restore_original,
                    preview_sound=preview_sound,
                )
            )
            if not isinstance(run_result, ContentAppCompleted):
                ask.ask_error(run_result.message)
                return False

            editor_result = run_result.result
            if isinstance(editor_result, EditorSaveFailed):
                ask.ask_error(editor_result.error)
                return False
            if isinstance(editor_result, WordSubstitutionPreviewRequested):
                staged = dict(editor_result.staged_items)
                restore_original = editor_result.original
                preview_result = run_tts_preview_app(
                    state,
                    make_word_substitution_preview_prompt(
                        editor_result.original,
                        editor_result.substitution,
                    ),
                )
                preview_sound = (
                    preview_result.sound if preview_result.completed else None
                )
                continue
            # Preserve the existing menu behavior: any normal editor close
            # (saved or unchanged) returns to the parent Text menu.
            return True

    @staticmethod
    def word_substitutions_menu(state: State) -> None:

        def on_inspect(_, __) -> None:
            normed_lang = language_util.normalize_language_code(state.project.language_code)
            subheading = UNCOMMON_WORDS_DESC.replace("%1", normed_lang)
            MenuUtil.print_screen_heading(state, "Uncommon words", subheading=subheading)

            # Make list of project text words (unfiltered, still including whitespace)
            all_words_raw = []
            for group in state.project.phrase_groups:
                for phrase in group.phrases:
                    all_words_raw.extend(phrase.words)

            items = app_text.get_uncommon_words(all_words_raw)
            if not items:
                printt("None found")
            else:
                for i in range(0, min(len(items), 25)):
                    item = items[i]
                    num_str = f"{COL_DIM}{str(item[1]).rjust(3)}"
                    instances_str = f"{COL_DEFAULT}{' | '.join(item[2])}"
                    print(f"{num_str}  {instances_str}")
            printt()
            ask.ask_enter_to_continue()

        def items_maker(_) -> list[MenuItem]:
            items = [
                MenuItem(
                    make_edit_substitutions_label,
                    lambda _, __: TextMenu.edit_word_substitutions(state),
                )
            ]
            # Only first-class languages (with a common-words dictionary) have
            # anything to inspect; supports_language handles code normalization
            if Whitelist.supports_language(state.project.language_code) and state.project.phrase_groups:
                items.append(MenuItem("Inspect project text for uncommon words", on_inspect))
            return items

        MenuUtil.menu(
            state,
            heading=make_subst_label,
            items=items_maker,
            subheading=SUBSTITUTIONS_DESC,
            breadcrumb="Word substitutions",
        )



# ---

def on_select_import(state: State, item: MenuItem) -> bool:

    # Print pseudo-menu heading
    match item.data:
        case "text":
            heading = "Import from text file"
        case "epub":
            heading = "Import from EPUB file"
        case "manual":
            heading = "Manually enter text"
        case _:
            raise Exception("Bad value")
    MenuUtil.print_screen_heading(state, heading)

    num_files = state.project.sound_segments.num_generated()
    num_markers = len(state.project.markers)
    num_substitutions = len(state.project.word_substitutions)
    if num_files > 0 or num_markers > 0 or num_substitutions > 0:
        # Confirm before discarding text-dependent project settings
        parts: list[str] = []
        if num_files > 0:
            files_noun = make_noun("file", "files", num_files)
            parts.append(
                f"{COL_ERROR}{num_files}{COL_DEFAULT} generated sound segment {files_noun}"
            )
        if num_markers > 0:
            marker_label = app_text.get_section_marker_label(
                state.project, is_title_case=False, is_singular=(num_markers == 1)
            )
            parts.append(f"{COL_ERROR}{num_markers}{COL_DEFAULT} {marker_label}")
        if num_substitutions > 0:
            substitution_label = make_noun(
                "word substitution", "word substitutions", num_substitutions
            )
            parts.append(
                f"{COL_ERROR}{num_substitutions}{COL_DEFAULT} {substitution_label}"
            )
        s = f"Replacing project text will discard {' and '.join(parts)}.\n"
        s += "Are you sure? "
        if not ask.ask_confirm(s):
            return False

    epub_path = ""
    epub_import_result: EpubImportResult | None = None

    match item.data:

        case "text":
            phrase_groups, raw_text, title = ask_phrase_groups.get_from_text_file(
                state.project.max_words,
                state.project.segmentation_strategy,
                pysbd_language=state.project.language_code,
                prefs=state.prefs,
                dialog_segmentation=state.project.dialog_segmentation
            )
            if not phrase_groups:
                return False

        case "manual":
            phrase_groups, raw_text = ask_phrase_groups.get_from_std_in(
                state.project.max_words, state.project.segmentation_strategy, pysbd_language=state.project.language_code,
                dialog_segmentation=state.project.dialog_segmentation
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
                dialog_segmentation=state.project.dialog_segmentation,
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

        case _:
            raise ValueError(f"Bad value: {item.data!r}")

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
            dialog_segmentation=state.project.dialog_segmentation,
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
            dialog_segmentation=state.project.dialog_segmentation,
            raw_text=raw_text,
            title=title,
            text_source_kind=text_source_kind,
        )

    if not state.real_time.custom_phrase_groups:
        state.real_time.project_text_line_range = None

    num_text_lines = len(phrase_groups)
    if epub_import_result:
        raw_text_path = os.path.join(state.project.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        raw_text_link = text_util.make_terminal_hyperlink(raw_text_path, raw_text_path, is_file=True)
        printt(f"{COL_ACCENT}A plain-text conversion{COL_DEFAULT} of the EPUB file was also saved here:")
        printt(f"{raw_text_link}")
        printt()

    printt(f"{COL_ACCENT}Segmented {num_text_lines} text lines{COL_DEFAULT} using the following settings:")
    segmentation_settings = state.project.book.segmentation_settings
    printt(f"- Text segmenter language code: {COL_ACCENT}{segmentation_settings.language_code or 'none'}")
    if segmentation_settings.max_words_per_segment:
        printt(f"- Text segmenter max_words_per_segment: {COL_ACCENT}{segmentation_settings.max_words_per_segment}")
    printt(f"- Text segmenter strategy: {COL_ACCENT}{segmentation_settings.strategy.label}")
    printt(f"- Dialog segmentation: {COL_ACCENT}{segmentation_settings.dialog_segmentation}")
    if segmentation_settings.dialog_segmentation:
        dialog_segment_count = sum(
            group.voice_index == DIALOG_VOICE_INDEX
            for group in phrase_groups
        )
        printt(
            f"  Text segments preassigned to voice sample 2: "
            f"{COL_ACCENT}{dialog_segment_count}"
        )
        if (
            dialog_segment_count
            and state.project.voice_select_mode != VoiceSelectMode.USER_DEFINED
        ):
            printt(
                f"{COL_DIM}  Voice selection mode must be set to: user-defined{COL_DEFAULT}"
            )
    printt()

    if state.project.language_code in ("en", "es"):
        hints.show_hint_if_necessary(
            state.prefs,
            HINT_TOLERANCE_FIRST_CLASS
        )
    if state.project.dialog_segmentation:
        hints.show_hint_if_necessary(
            state.prefs,
            HINT_DIALOG_VOICE,
        )

    ask.ask_enter_to_continue()

    return False

def on_ask_max_size(state: State, _) -> None:

    MenuUtil.print_screen_heading(state, "Max words per segment")

    printt("On text import, this is the maximum number of words to be used for a single text segment.")
    printt(f"Recommended range for current model: {COL_ACCENT}{TtsModelType.recommended_range_string(Tts.get_type().value)}")
    printt()

    ask.ask_number_and_save(
        state.project,
        attr="max_words",
        prompt="Enter max words per segment:",
        min_value=MAX_WORDS_PER_SEGMENT_MIN, max_value=MAX_WORDS_PER_SEGMENT_MAX,
        default_value=MAX_WORDS_PER_SEGMENT_DEFAULT,
        success_prefix="Max words per segment set to:",
        is_int=True
    )

def make_edit_substitutions_label(state: State) -> str:
    count = len(state.project.word_substitutions)
    return f"Edit word substitutions {make_currently_string(count, value_prefix='items: ')}"


def make_subst_label(state: State) -> str:
    num_subst = len(state.project.word_substitutions)
    if num_subst > 0:
        value = f"{num_subst} {make_noun('item', 'items', num_subst)}" if num_subst > 0 else "none"
        label = f"Word substitutions {make_currently_string(value)}"
    else:
        label = f"Word substitutions {COL_DIM}(optional)"
    return label


SEG_SUBHEADING = \
"""On import, text will be segmented into sentences and phrases using the settings
shown below. Project language code can also affect how the text is segmented.
"""

DIALOG_SEGMENTATION_DESC = \
f"""Detects and splits dialog from surrounding narration,
making it easier to assign different voices.
For example, {COL_DIM}He said, "Hello."{COL_DEFAULT} becomes {COL_DIM}He said,{COL_DEFAULT} and {COL_DIM}"Hello."{COL_DEFAULT}
Dialog segments are preassigned to voice sample 2.

For single-voice narration, leave this off to preserve natural flow.
"""

SUBSTITUTIONS_DESC = \
f"""Word pairs that get swapped into the text before it is sent to the TTS model.
Good for fixing mispronounced names and unusual words.

{COL_DIM}Eg, mapping "Nguyen" to "win" so the model pronounces it correctly.
Works best with a bit of experimentation.
"""

UNCOMMON_WORDS_DESC = \
f"""Words in the project text not found in the app's
%1 \"common words\" dictionary, sorted by frequency.
"""

