from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.phrase_group_ask_util import PhraseGroupAskUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class TextMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_heading(_) -> str:
            if state.project.phrase_groups:
                s = str(len(state.project.phrase_groups))
                s += " line" if len(state.project.phrase_groups) == 1 else " lines"
                currently = make_currently_string(s)
            else:
                currently = f"{COL_ERROR}(required)"
            return f"Text {currently}"

        def make_max_size_label(_) -> str:
            value = make_parameter_value_string(
                state.project.max_words, MAX_WORDS_PER_SEGMENT_DEFAULT
            )
            return make_menu_label("Text segmentation max words per segment", value)

        def on_print(_: State, __: MenuItem) -> None:
            AppUtil.print_project_text(
                phrase_groups=state.project.phrase_groups,
                extant_indices = set( state.project.sound_segments.sound_segments_map.keys() ),
                language_code_used=state.project.applied_language_code,
                max_words_used=state.project.applied_max_words,
                strategy_used=state.project.applied_strategy or list(SegmentationStrategy)[0]
            )
            AskUtil.ask_enter_to_continue()

        items = [
            MenuItem("Import from text file", on_set_text, data="import"),
            MenuItem("Manually input/paste text", on_set_text, data="manual"),
            MenuItem("Print text segments", on_print),
            MenuItem(make_max_size_label, on_ask_max_size),
            MenuItem(
                lambda _: make_menu_label("Text segmentation strategy", state.project.segmentation_strategy.label.lower()),
                lambda _, __: TextMenu.strategy_menu(state)
            )
        ]
        MenuUtil.menu(state, make_heading, items, hint=HINT_LINE_BREAKS)

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
            subheading=SEG_STRATEGY_SUBHEADING
        )

def on_set_text(state: State, item: MenuItem) -> bool:

    num_files = state.project.sound_segments.num_generated()
    if num_files > 0:
        # First confirm
        s = f"Replacing project text will invalidate all {num_files} previously generated sound segment files for this project.\n"
        s += "Are you sure? "
        if not AskUtil.ask_confirm(s):
            return False

    if item.data == "import":
        phrase_groups, raw_text = PhraseGroupAskUtil.get_from_text_file(
            state.project.max_words, 
            state.project.segmentation_strategy, 
            pysbd_language=state.project.language_code,
            prefs=state.prefs
        )
        if not phrase_groups:
            return False
    elif item.data == "manual":
        phrase_groups, raw_text = PhraseGroupAskUtil.get_from_std_in(
            state.project.max_words, state.project.segmentation_strategy, pysbd_language=state.project.language_code
        )
        if not phrase_groups:
            print_feedback("Cancelled")
            return False
    else:
        return False
    
    # Preview
    AppUtil.print_project_text(
        phrase_groups=phrase_groups,
        extant_indices=None,
        language_code_used=state.project.language_code,
        max_words_used=state.project.max_words,
        strategy_used=state.project.segmentation_strategy
    )

    # Confirm
    if not AskUtil.ask_confirm():
        print_feedback("Cancelled")
        return False

    # Delete now-outdated gens
    state.project.sound_segments.delete_all()

    # Commit
    state.project.set_phrase_groups_and_save(
        phrase_groups=phrase_groups,
        strategy=state.project.segmentation_strategy,
        max_words=state.project.max_words,
        language_code=state.project.language_code,
        raw_text=raw_text
    )

    if not state.real_time.custom_phrase_groups:
        state.real_time.line_range = None

    print_feedback("Project text has been set")
    return True

def on_ask_max_size(state: State, _) -> None:

    print_heading("Max words per segment")

    printt("This is the maximum number of words that will be used for a single TTS prompt.")
    printt("This value is applied at the point text is imported into the project.")
    printt(f"Recommended range for current model: {COL_ACCENT}{TtsModelInfos.recommended_range_string(Tts.get_type().value)}")
    printt()

    prompt = f"Enter max words per segment {COL_DIM}(between {MAX_WORDS_PER_SEGMENT_MIN}-{MAX_WORDS_PER_SEGMENT_MAX}){COL_DEFAULT}:"
    AskUtil.ask_number(
        state.project,
        prompt=prompt,
        lb=MAX_WORDS_PER_SEGMENT_MIN, ub=MAX_WORDS_PER_SEGMENT_MAX,
        attr="max_words",
        success_prefix="Max words per segment set to:",
        is_int=True
    )

SEG_STRATEGY_SUBHEADING = \
"""When text is imported to the project, this dictates how it will be segmented 
for text-to-speech inference.
"""