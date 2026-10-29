from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.app_types import SectionMarkerMode, ExportType, HighShelfEq, MinSegmentSentenceSilence, NormalizationType, RoomReverb, RoomReverbStereo, RoomTone
from tts_audiobook_tool import ask
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.app_types.output_range_info import OutputRangeInfo
from tts_audiobook_tool.menus.section_markers_menu import SectionMarkersMenu
from tts_audiobook_tool.menus.menu_shared import make_concat_menu_heading, make_full_output_files_subheading, make_output_files_subheading, make_output_range_info_strings
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.menu_util import MAX_AUTO_HOTKEY_MENU_ITEMS, MenuItem, MenuUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool import text_util
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.voice_spatial_policy import VoiceSpatialPolicy
from tts_audiobook_tool.app_types.voice import stereo_position_label
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.system_support.browser import (
    get_chromium_info,
    launch_player_with_chromium,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *

class ConcatMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_start_label(state: State) -> str:
            s = "Start"
            num_generated = state.project.sound_segments.num_generated()
            if num_generated == 0:
                s += f" {COL_DIM}({COL_ERROR}requires generated audio{COL_DIM})"
            return s

        def make_section_markers_label(_: State) -> str:
            qty = len(state.project.markers)
            label = "Section markers "
            if qty > 0:
                noun = make_noun('item', 'items', qty)
                mode = state.project.chapter_mode.label.lower()
                value = f"{qty} {noun}{COL_DIM}, mode: {COL_ACCENT}{mode}"
                label += make_currently_string(value)
            else:
                label += f"{COL_DIM}(optional)"
            return label

        def make_split_points_label(_: State) -> str:
            num_markers = len(state.project.markers)
            if num_markers > 0:
                items_noun = make_noun('item', 'items', num_markers)
                num_files = len(make_file_line_ranges(state.project.markers, len(state.project.phrase_groups)))
                files_noun = make_noun('file', 'files', num_files)
                value = make_currently_string(f"{num_markers} {items_noun} = {num_files} {files_noun}")
            else:
                value = f"{COL_DIM}(optional)"
            label = f"Split points {value}"
            return label

        def make_items(_: State) -> list[MenuItem]:

            file_type_value = state.project.export_type.label
            if state.project.export_type == ExportType.AAC:
                file_type_value += f" {COL_DIM}{state.prefs.aac_bitrate}"

            items = []

            items.append(
                MenuItem(make_start_label, lambda _, __: ask_output_indices_and_make(state))
            )

            is_limited = state.project.has_multiple_book_sections()
            label_function = make_split_points_label if is_limited else make_section_markers_label
            section_markers_action = (
                SectionMarkersMenu.edit_section_markers
                if is_limited
                else SectionMarkersMenu.menu
            )
            items.append(
                MenuItem(label_function, lambda _, __: section_markers_action(state))
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("File type", file_type_value),
                    lambda _, __: ConcatMenu.file_type_menu(state),
                    superlabel="Options"
                )
            )

            items.append(
                MenuItem(
                    "Pause modulation rules",
                    lambda _, __: ConcatMenu.prosody_modulation_menu(state)
                )
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("Section break sound effect", state.project.use_break_sound_effect),
                    lambda _, __: ConcatMenu.section_break_menu(state)
                )
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("Reader uses phrase subdivisions", state.project.subdivide_phrases),
                    lambda _, __: ConcatMenu.subdivide_menu(state)
                )
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label(
                        label="Loudness normalization",
                        value=state.project.normalization_type.value.label,
                    ),
                    lambda _, __: ConcatMenu.normalization_menu(state),
                    superlabel="Post-processing"
                )
            )

            label = make_menu_label(label="Generative upsampling", value=state.project.use_upsampler)
            if Tts.get_model_support(state.project).get_output_sample_rate(state.project) >= 44_100 and state.project.use_upsampler:
                label += f"{COL_ERROR}*"
            items.append(
                MenuItem(
                    label,
                    lambda _, __: ConcatMenu.upsample_menu(state)
                )
            )

            # Room reverb itself is a per-voice setting. The stereo effect is
            # project-wide and applies whenever the voice in use has reverb on.
            items.append(
                MenuItem(
                    lambda _: make_menu_label(ROOM_REVERB_STEREO_TITLE, state.project.room_reverb_stereo.label),
                    lambda _, __: ConcatMenu.room_reverb_stereo_menu(state)
                )
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label(ROOM_TONE_TITLE, state.project.room_tone.label),
                    lambda _, __: ConcatMenu.room_tone_menu(state)
                )
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label(
                        label="Treble EQ",
                        value=state.project.get_high_shelf().id,
                    ),
                    lambda _, __: ConcatMenu.high_shelf_menu(state)
                )
            )

            can_and_haz = chromium_info and ProjectUtil.get_latest_concat_files(state.project, limit=1)
            if can_and_haz:
                items.append(
                    MenuItem(
                        f"Open audiobook file in the player app",
                        lambda _, __: ConcatMenu.open_audiobook_menu(state),
                        superlabel=" ", superlabel_no_blank_line=True # yes rly
                    )
                )

            return items

        MenuUtil.menu(state, make_concat_menu_heading, make_items, subheading=make_output_files_subheading)

    @staticmethod
    def file_type_menu(state: State) -> None:

        def on_select(value: ExportType) -> None:
            state.project.export_type = value
            state.project.save()
            print_feedback(f"File type set to: {value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text="File type",
            labels=[item.label for item in list(ExportType)],
            values=[item for item in list(ExportType)],
            current_value=state.project.export_type,
            default_value=list(ExportType)[0],
            on_select=on_select
        )

    @staticmethod
    def prosody_modulation_menu(state: State) -> None:

        def make_items(_: State) -> list[MenuItem]:
            return [
                MenuItem(
                    lambda _: make_menu_label(
                        "Pauses between segments",
                        state.project.reason_pauses.menu_label,
                    ),
                    lambda _, __: ConcatMenu.reason_pauses_menu(state),
                ),
                MenuItem(
                    make_limit_silence_gaps_label,
                    lambda _, __: ConcatMenu.limit_silence_gaps_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label(
                        MIN_SEGMENT_SENTENCE_SILENCE_TITLE,
                        state.project.min_segment_sentence_silence.label,
                    ),
                    lambda _, __: ConcatMenu.min_segment_sentence_silence_menu(state),
                ),
            ]

        MenuUtil.menu(
            state=state,
            heading="Pause modulation rules",
            items=make_items,
        )

    @staticmethod
    def min_segment_sentence_silence_menu(state: State) -> None:

        def on_select(value: MinSegmentSentenceSilence) -> None:
            state.project.min_segment_sentence_silence = value
            state.project.save()
            print_feedback(f"Value set to: {value.label}")

        item = MenuUtil.make_cycle_item(
            label="Value",
            values=list(MinSegmentSentenceSilence),
            value_labels=[v.label for v in MinSegmentSentenceSilence],
            get_value=lambda: state.project.min_segment_sentence_silence,
            default_value=MinSegmentSentenceSilence.get_default(),
            on_select=on_select,
        )

        MenuUtil.menu(
            state=state,
            heading=MIN_SEGMENT_SENTENCE_SILENCE_TITLE,
            items=[item],
            subheading=MIN_SEGMENT_SENTENCE_SILENCE_SUBHEADING,
            breadcrumb=MIN_SEGMENT_SENTENCE_SILENCE_TITLE,
        )

    @staticmethod
    def reason_pauses_menu(state: State) -> None:

        def on_select(value: ReasonPauseTypes) -> None:
            state.project.reason_pauses = value.value
            state.project.save()
            print_feedback(f"Pauses between segments set to: {value.value.label}")

        current_value = ReasonPauseTypes.get_by_id(state.project.reason_pauses.id)

        MenuUtil.options_menu(
            state=state,
            heading_text="Pauses between segments",
            labels=[item.value.label for item in ReasonPauseTypes],
            values=list(ReasonPauseTypes),
            current_value=current_value,
            default_value=ReasonPauseTypes.default(),
            on_select=on_select,
            sublabels=[item.value.description for item in ReasonPauseTypes],
            subheading=REASON_PAUSES_SUBHEADING,
        )

    @staticmethod
    def normalization_menu(state: State) -> None:

        def on_select(value: NormalizationType) -> None:
            state.project.normalization_type = value
            state.project.save()
            print_feedback(f"Normalization set to: {value.value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Loudness normalization",
            labels=[item.value.label for item in list(NormalizationType)],
            values=[item for item in list(NormalizationType)],
            current_value=state.project.normalization_type,
            default_value=list(NormalizationType)[0],
            on_select=on_select,
            sublabels=[item.value.description for item in NormalizationType],
            subheading=LOUDNORM_SUBHEADING,
        )

    @staticmethod
    def high_shelf_menu(state: State) -> None:

        def on_select(value: HighShelfEq) -> None:
            state.project.high_shelf = value.id
            state.project.save()
            print_feedback(f"Treble EQ set to: {value.id}")

        current = state.project.get_high_shelf()

        MenuUtil.options_menu(
            state=state,
            heading_text="Treble EQ",
            labels=[item.id.capitalize() for item in list(HighShelfEq)],
            values=[item for item in list(HighShelfEq)],
            current_value=current,
            default_value=PROJECT_DEFAULT_HIGH_SHELF_EQ,
            on_select=on_select,
            subheading=HIGH_SHELF_SUBHEADING
        )

    @staticmethod
    def limit_silence_gaps_menu(state: State) -> None:

        def on_enabled_select(value: bool) -> None:
            state.project.limit_silence_gaps = value
            state.project.save()
            print_feedback(f"{LIMIT_SILENCE_GAPS_TITLE} set to: {'enabled' if value else 'disabled'}")

        def make_items(_: State) -> list[MenuItem]:
            items = [
                MenuUtil.make_cycle_item(
                    label="Value",
                    values=[False, True],
                    value_labels=["disabled", "enabled"],
                    get_value=lambda: state.project.limit_silence_gaps,
                    default_value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS,
                    on_select=on_enabled_select,
                )
            ]
            # Duration only applies when enabled
            if state.project.limit_silence_gaps:
                items.append(
                    MenuUtil.make_number_item(
                        state=state,
                        target="limit_silence_gaps_duration",
                        base_label="Duration",
                        default_value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
                        is_minus_one_default=False,
                        num_decimals=2,
                        prompt="Enter maximum silence duration (seconds):",
                        min_value=SILENCE_GAP_DURATION_MIN,
                        max_value=SILENCE_GAP_DURATION_MAX,
                        value_suffix="s",
                    )
                )
            return items

        MenuUtil.menu(
            state=state,
            heading=LIMIT_SILENCE_GAPS_TITLE,
            items=make_items,
            subheading=LIMIT_SILENCE_GAPS_SUBHEADING,
            breadcrumb=LIMIT_SILENCE_GAPS_TITLE,
        )

    @staticmethod
    def upsample_menu(state: State) -> None:

        lava_sr_available, _ = ModelWorker.probe_lava_sr_blocking()

        def on_select(value: bool) -> None:
            if value and not lava_sr_available:
                ask.ask_error(
                    "LavaSR v2 is not installed; generative upsampling cannot be enabled",
                )
                return
            state.project.use_upsampler = value
            state.project.save()
            print_feedback(f"Generative upsampling set to: {value}")

        if not lava_sr_available:
            subheading = f"{Ansi.ITALICS}LavaSR v2 upsampler not installed"
        else:
            subheading = UPSAMPLE_SUBHEADING
            link = text_util.make_terminal_hyperlink(LAVA_SR_PROJECT_URL, "LavaSR")
            subheading = subheading.replace("%1", link)
            sample_rate = Tts.get_model_support(state.project).get_output_sample_rate(state.project)
            if sample_rate >= 44_100:
                subheading += (
                    "\n\n"
                    f"{COL_ERROR}*{COL_DEFAULT} {state.project.get_tts_model_type().value.ui['short_name']} already outputs audio at a samplerate of {sample_rate}."
                )

        MenuUtil.options_menu(
            state=state,
            heading_text="Generative upsampling",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.use_upsampler,
            default_value=PROJECT_DEFAULT_GENERATIVE_UPSAMPLING,
            on_select=on_select
        )

    @staticmethod
    def subdivide_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.subdivide_phrases = value
            state.project.save()
            print_feedback(f"Reader phrase subdivisions set to: {state.project.subdivide_phrases}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Subdivide into phrases",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.subdivide_phrases,
            default_value=False,
            on_select=on_select,
            subheading=SUBDIVIDE_SUBHEADING,
        )

    @staticmethod
    def room_tone_menu(state: State) -> None:

        def on_select(value: RoomTone) -> None:
            state.project.room_tone = value
            state.project.save()
            print_feedback(f"{ROOM_TONE_TITLE} set to: {value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text=ROOM_TONE_TITLE,
            labels=[item.label for item in RoomTone],
            values=list(RoomTone),
            current_value=state.project.room_tone,
            default_value=RoomTone.get_default(),
            on_select=on_select,
            sublabels=[
                f"{item.level_db:g} dBFS" if item.level_db is not None else "No added noise"
                for item in RoomTone
            ],
            subheading=ROOM_TONE_SUBHEADING,
        )

    @staticmethod
    def room_reverb_stereo_menu(state: State) -> None:

        def on_select(value: RoomReverbStereo) -> None:
            state.project.room_reverb_stereo = value
            state.project.save()
            print_feedback(f"{ROOM_REVERB_STEREO_TITLE} set to: {value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text=ROOM_REVERB_STEREO_TITLE,
            labels=[item.label for item in RoomReverbStereo],
            values=list(RoomReverbStereo),
            current_value=state.project.room_reverb_stereo,
            default_value=RoomReverbStereo.get_default(),
            on_select=on_select,
            subheading=ROOM_REVERB_STEREO_SUBHEADING,
        )

    @staticmethod
    def section_break_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.use_break_sound_effect = value
            state.project.save()
            print_feedback(f"Set to: {value}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Section break sound effect",
            subheading=SECTION_BREAK_SUBHEADING,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.use_break_sound_effect,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def open_audiobook_menu(state: State) -> None:

        if chromium_info is None:
            return

        file_infos = ProjectUtil.get_latest_concat_files(state.project, MAX_AUTO_HOTKEY_MENU_ITEMS)
        if not file_infos:
            print_feedback("No files found")
            return

        user_data_dir = app_paths.get_chromium_user_data_dir()

        def on_item(_: State, item: MenuItem) -> None:
            assert(isinstance(chromium_info, tuple))
            launch_player_with_chromium(
                chromium_path=chromium_info[1],
                audio_file_path=item.data,
                user_data_dir=user_data_dir
            )

        def make_items(_: State) -> list[MenuItem]:
            items = []
            for file_path, modified_date in file_infos:
                label = f"{Path(file_path).name}\n      {COL_DIM}{modified_date}{COL_DEFAULT}"
                item = MenuItem(label, on_item, file_path)
                items.append(item)
            return items

        subheading = OPEN_AUDIOBOOK_SUBHEADING
        subheading = subheading.replace("%1", chromium_info[0])
        subheading = subheading.replace("%2", user_data_dir)

        MenuUtil.menu(
            state=state,
            heading="Open audiobook file",
            items=make_items,
            subheading=subheading,
            one_shot=True
        )

def make_enabled_option_confirmation_lines(state: State) -> list[str]:
    """
    Lists enabled post-processing options, followed by the output channel format.
    """
    lines = []

    def add(prefix: str, value: object) -> None:
        lines.append(f"  {COL_DIM}- {prefix}: {COL_DEFAULT}{value}")

    # Same order as the menu
    reason_pauses = state.project.reason_pauses
    if reason_pauses.id != ReasonPauseTypes.default().value.id:
        add("Pauses between segments", reason_pauses.menu_label)
    if state.project.limit_silence_gaps:
        add(
            "Maximum silence inside segments",
            f"True {state.project.limit_silence_gaps_duration:.2f}s",
        )
    if state.project.min_segment_sentence_silence != MinSegmentSentenceSilence.DISABLED:
        add(MIN_SEGMENT_SENTENCE_SILENCE_TITLE, True)
    if state.project.use_break_sound_effect:
        add("Section break sound effects", True)
    if state.project.normalization_type != NormalizationType.DISABLED:
        add("Loudness normalization", state.project.normalization_type.value.label)
    if state.project.use_upsampler:
        add("Generative upsampling", True)
    spatial_policy = VoiceSpatialPolicy.from_project(state.project)
    stereo_position = spatial_policy.stereo_position_summary  # None: differs between voices
    if stereo_position != 0:
        add(
            STEREO_POSITION_TITLE,
            stereo_position_label(stereo_position) if stereo_position is not None else "Varies by voice",
        )
    room_reverb = spatial_policy.room_reverb_summary  # None: differs between voices
    if room_reverb != RoomReverb.OFF:
        add(ROOM_REVERB_TITLE, room_reverb.label if room_reverb is not None else "Varies by voice")
        if state.project.room_reverb_stereo != RoomReverbStereo.MONO:
            add(ROOM_REVERB_STEREO_TITLE, state.project.room_reverb_stereo.label)
    if state.project.room_tone != RoomTone.OFF:
        add(ROOM_TONE_TITLE, state.project.room_tone.label)
    high_shelf = state.project.get_high_shelf()
    if high_shelf != HighShelfEq.DISABLED:
        add("Treble EQ", high_shelf.id.capitalize())
    channels = spatial_policy.requested_channels
    add("Channel output", "Stereo" if channels == 2 else "Mono")
    return lines


def print_enabled_option_confirmation_lines(state: State) -> None:
    """
    Prints enabled options and output channel format, preceded by a blank line.
    """
    lines = make_enabled_option_confirmation_lines(state)
    if not lines:
        return
    printt()
    for line in lines:
        printt(line)


def ask_output_indices_and_make(state: State) -> None:

    num_generated = state.project.sound_segments.num_generated()
    if not state.prefs.project_dir or num_generated == 0:
        ask.ask_error("Requires generated audio")
        return

    type_string = "AAC/M4B" if state.project.export_type == ExportType.AAC else "FLAC"

    MenuUtil.print_screen_heading(state, "Start", subheading=make_full_output_files_subheading)

    should_ask_file_numbers = (state.project.chapter_mode == SectionMarkerMode.FILES) and len(state.project.markers) > 0
    if should_ask_file_numbers:

        infos = OutputRangeInfo.make_output_range_infos(state.project)

        prefill = ProjectUtil.get_concat_files_range_string(state.project)
        result = ask_output_indices(infos, prefill=prefill)
        if result is None:
            return
        output_indices, range_string = result
        bookmark_indices = []
        ProjectUtil.persist_concat_files_range_string(state.project, range_string)

        noun = make_noun('file', 'files', len(output_indices))
        s = f"Will create the following {type_string} {noun}:"
        print_accent_separator(s, and_blank_line=False)

        strings = make_output_range_info_strings(infos, output_indices)
        s = "\n".join( ("    " + item) for item in strings)
        printt(s)
        print_enabled_option_confirmation_lines(state)
        printt()

        b = ask.ask_confirm()
        if not b:
            return

    else:

        output_indices = []
        if state.project.can_use_bookmark_section_markers():
            bookmark_indices = sorted(state.project.markers)
        else:
            bookmark_indices = []

        info = OutputRangeInfo.make_single_info(state.project)
        s = f"Will create a single {type_string} file:"
        print_accent_separator(s, and_blank_line=False)
        s = f"    All lines "
        s += f"({info.num_files_exist}/{info.num_segments} generated){COL_DEFAULT}"
        printt(s)
        print_enabled_option_confirmation_lines(state)
        printt()

        b = ask.ask_confirm()
        if not b:
            return

    ConcatUtil.make_files(
        state=state,
        file_cut_indices=output_indices,
        bookmark_indices=bookmark_indices
    )

def ask_output_indices(infos: list[OutputRangeInfo], prefill: str = "") -> tuple[list[int], str] | None:
    """
    Returns the selected zero-indexed output file indices, excluding files
    with no generated audio, and the input normalized as a range string
    (including those files), or None on cancel/error.
    Out-of-range file numbers are ignored.
    """

    printt("Enter file numbers to create:")
    printt(f"{COL_DIM}(For example: \"1, 2, 4\" or  \"2-5\", or \"all\")")
    inp = ask.ask_input(prefill=prefill)

    if not inp:
        return None

    # Lenient: out-of-range values are dropped; only syntax errors are reported
    selected_indices, warnings = RangeStringUtil.parse_ranges_string(inp, len(infos), strict=False)
    if warnings:
        message = "\n".join(warnings)
        ask.ask_error(message)
        return None

    # Files without generated audio are skipped rather than rejected
    indices = sorted(index for index in selected_indices if infos[index].num_files_exist > 0)
    if not indices:
        ask.ask_error("No generated audio for the selected files")
        return None

    range_string = RangeStringUtil.make_ranges_string(selected_indices, len(infos))
    return indices, range_string

chromium_info = get_chromium_info()

def make_limit_silence_gaps_label(state: State) -> str:
    if state.project.limit_silence_gaps:
        value = f"enabled {state.project.limit_silence_gaps_duration:.2f}s"
    else:
        value = "disabled"
    return make_menu_label(
        label="Maximum silence inside segments",
        value=value,
        default=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS
    )

LOUDNORM_SUBHEADING = \
"""Applies a final pass after concatenation to standardize overall loudness and
control peaks. Choose Stronger for a louder, more consistent presentation on
mobile devices or in noisy environments."""

SUBDIVIDE_SUBHEADING = \
"""Affects text highlighting in the player/reader app.
When False, highlighted text maps directly to the TTS prompts used to generate the sound segments.
When True, highlighted text is further sub-segmented by phrase (Requires \"speech-to-text validation\"
to be enabled during TTS sound generation)."""

SECTION_BREAK_SUBHEADING = \
"""When set to true, inserts a subtle \"page turn\" sound effect when
two or more blank lines are encountered in the text. This can be a
useful audible cue, so long as the text is formatted for it.
Also inserts a more prominent sound at section breaks."""

OPEN_AUDIOBOOK_SUBHEADING = \
f"""Select file to be opened in the player app using %1, which will be
launched with a dedicated user profile and the following flags
to enable opening local audio files without user input:
  {COL_DIM}--allow-file-access-from-files
  --autoplay-policy=no-user-gesture-required
  --user-data-dir=%2"""

HIGH_SHELF_SUBHEADING = \
"""Applies a high-shelf equalizer pass to compensate for dull or muffled-sounding TTS output.
May be useful for lower-fidelity TTS models.
Is also applied to realtime playback."""

LIMIT_SILENCE_GAPS_TITLE = "Maximum silence inside segments"

LIMIT_SILENCE_GAPS_SUBHEADING = \
"""Limits instances of silence within sound segment from extending beyond
a certain duration.
Also applies to realtime playback.

Large values can be used to prevent long pauses (eg, 1s-2s);
small values can be used to influence pacing and prosody (eg, 0s-0.3s).
(Consider using \"Text segmentation strategy: Sentence.\" in that case)"""

MIN_SEGMENT_SENTENCE_SILENCE_TITLE = "Minimum sentence silence inside segments"

MIN_SEGMENT_SENTENCE_SILENCE_SUBHEADING = \
f"""Enforces a minimum amount of silence between sentences within a sound segment ({MinSegmentSentenceSilence.ENABLED.duration:g}s).
Requires audio to have been generated using STT validation.
Also applies to realtime playback."""

ROOM_TONE_TITLE = "Room tone"

ROOM_TONE_SUBHEADING = \
"""Adds subtle room tone under the audio, preventing unnatural digital silence. ACX-recommended practice.
Is also applied to realtime playback."""

STEREO_POSITION_TITLE = "Stereo position"

ROOM_REVERB_TITLE = "Room reverb"

ROOM_REVERB_STEREO_TITLE = "Room reverb stereo effect"

ROOM_REVERB_STEREO_SUBHEADING = \
"""Spreads room reverb across left and right channels, for a wider sense of space.
Applies only to voices that have room reverb set (see voice sample submenu)."""

REASON_PAUSES_SUBHEADING = \
"""Controls the silence inserted between generated segments."""

UPSAMPLE_SUBHEADING = \
"""Uses %1 to enhance speech and generate higher-frequency detail,
producing 48 kHz audio.

FLAC file output is recommended to preserve the generated detail."""
