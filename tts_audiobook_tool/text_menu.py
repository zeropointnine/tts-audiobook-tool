from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.voice_menu_shared import VoiceMenuShared

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
            value = VoiceMenuShared.make_parameter_value_string(
                state.project.max_words, MAX_WORDS_PER_SEGMENT_DEFAULT
            )
            return make_menu_label("Text segmentation max words per segment", value)

        items = [
            MenuItem("Import from text file", on_set_text, data="import"),
            MenuItem("Manually enter/paste text", on_set_text, data="manual"),
            MenuItem("Print text segments", lambda _, __: AppUtil.print_project_text(state)),
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
        phrase_groups, raw_text = AppUtil.get_phrase_groups_from_ask_text_file(
            state.project.max_words, state.project.segmentation_strategy, pysbd_language=state.project.language_code
        )
    elif item.data == "manual":
        phrase_groups, raw_text = AppUtil.get_text_groups_from_ask_std_in(
            state.project.max_words, state.project.segmentation_strategy, pysbd_language=state.project.language_code
        )
    else:
        return False
    if not phrase_groups:
        return False

    # Print
    AppUtil.print_text_groups(phrase_groups)
    s = f"... is how the text has been segmented for inference"

    s += f"\n    (max words per segment: {COL_ACCENT}{int(state.project.max_words)}{COL_DEFAULT}, " \
        f"segmentation strategy: {COL_ACCENT}{state.project.segmentation_strategy.label}{COL_DEFAULT}, " \
        f"project language code: {COL_ACCENT}{state.project.language_code or 'none'}{COL_DEFAULT})"
    printt(s)
    printt()

    # Confirm
    if not AskUtil.ask_confirm():
        return False

    # Delete now-outdated gens
    old_sound_segments = state.project.sound_segments.sound_segments
    for path in old_sound_segments.values():
        delete_silently(path)

    # Commit
    state.project.set_phrase_groups_and_save(
        phrase_groups=phrase_groups,
        strategy=state.project.segmentation_strategy,
        max_words=state.project.max_words,
        language_code=state.project.language_code,
        raw_text=raw_text
    )

    if not state.real_time.custom_text_groups:
        state.real_time.line_range = None

    print_feedback("Project text has been set")
    return True

def on_ask_max_size(state: State, _) -> None:

    Hint.show_hint_if_necessary(state.prefs, HINT_SEG_MAX_SIZE)

    prompt = f"Enter max words per segment ({MAX_WORDS_PER_SEGMENT_MIN} <= value <= {MAX_WORDS_PER_SEGMENT_MAX}):"
    AskUtil.ask_number(
        state.project,
        prompt=prompt,
        lb=MAX_WORDS_PER_SEGMENT_MIN, ub=MAX_WORDS_PER_SEGMENT_MAX,
        attr_name="max_words",
        success_prefix="Max words per segment set to:",
        is_int=True
    )

SEG_STRATEGY_SUBHEADING = \
"""When text is imported to the project, this dictates how
it is segmented for text-to-speech inference.
"""