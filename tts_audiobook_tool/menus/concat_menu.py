import torch

from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.app_types import SectionMarkerMode, ExportType, HighShelfEq, NormalizationType
from tts_audiobook_tool import ask
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.app_types.output_range_info import OutputRangeInfo
from tts_audiobook_tool.menus.section_markers_limited_menu import SectionMarkersLimitedMenu
from tts_audiobook_tool.menus.section_markers_menu import SectionMarkersMenu
from tts_audiobook_tool.menus.menu_shared import make_output_files_subheading, make_output_range_info_strings
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.sound.sidon_util import SidonUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.system_support.browser import (
    get_chromium_info,
    launch_player_with_chromium,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
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
                num_files = num_markers + 1
                files_noun = make_noun('file', 'files', num_files)
                value = make_currently_string(f"{num_markers} {items_noun} = {num_files} {files_noun}")
            else:
                value = f"{COL_DIM}(optional)"
            label = f"File split points {value}"
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
            if is_limited:    
                items.append( 
                    MenuItem(make_split_points_label, lambda _, __: SectionMarkersLimitedMenu.menu(state)) 
                )
            else:
                items.append( 
                    MenuItem(make_section_markers_label, lambda _, __: SectionMarkersMenu.menu(state)) 
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
                    lambda _: make_menu_label("Reader phrase subdivision", state.project.subdivide_phrases), 
                    lambda _, __: ConcatMenu.subdivide_menu(state)
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
                    lambda _: make_menu_label("Loudness normalization", state.project.normalization_type.value.label),
                    lambda _, __: ConcatMenu.normalization_menu(state),
                    superlabel="Post-processing options"
                )
            )

            if torch.cuda.is_available() and SidonUtil.has_sidon():
                items.append(
                    MenuItem(
                        lambda _: make_menu_label("Generative upsampling", state.project.use_upsampler),
                        lambda _, __: ConcatMenu.upsample_menu(state)
                    )
                )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("Treble lift", state.project.get_high_shelf().id),
                    lambda _, __: ConcatMenu.high_shelf_menu(state)
                )
            )

            can_and_haz = chromium_info and ProjectUtil.get_latest_concat_files(state.project)
            if can_and_haz:
                items.append(
                    MenuItem(
                        f"Open audiobook file in the player app", 
                        lambda _, __: ConcatMenu.open_audiobook_menu(state),
                        superlabel=" ", superlabel_no_blank_line=True # yes rly
                    )
                )

            return items

        MenuUtil.menu(state, "Create audiobook file/s", make_items, subheading=make_output_files_subheading, breadcrumb="Create audiobook")

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
            subheading=LOUDNORM_SUBHEADING,
            hint=HINT_OUTE_LOUD_NORM if Tts.get_type() == TtsModelInfos.OUTE else None
        )

    @staticmethod
    def high_shelf_menu(state: State) -> None:

        def on_select(value: HighShelfEq) -> None:
            state.project.high_shelf = value.id
            state.project.save()
            print_feedback(f"Treble lift set to: {value.id}")

        current = state.project.get_high_shelf()

        MenuUtil.options_menu(
            state=state,
            heading_text="Treble lift",
            labels=[item.id.capitalize() for item in list(HighShelfEq)],
            values=[item for item in list(HighShelfEq)],
            current_value=current,
            default_value=HighShelfEq.DISABLED,
            on_select=on_select,
            subheading=HIGH_SHELF_SUBHEADING
        )

    @staticmethod
    def upsample_menu(state: State) -> None:
        
        def on_select(value: bool) -> None:
            state.project.use_upsampler = value
            state.project.save()
            print_feedback(f"Generative upsampling set to: {value}")

        if not torch.cuda.is_available():
            subheading = "Requires CUDA, which is not available on this system.\n"
        elif not SidonUtil.has_sidon():
            subheading = f"{Ansi.ITALICS}Sidon upsampler not installed\n"
        else: 
            subheading = UPSAMPLE_SUBHEADING

        MenuUtil.options_menu(
            state=state,
            heading_text="Generative upsampling",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.use_upsampler,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def subdivide_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.subdivide_phrases = value
            state.project.save()
            print_feedback(f"Reader phrase subdivision set to: {state.project.subdivide_phrases}")

        MenuUtil.options_menu(
            state=state,
            heading_text="Subdivide into phrases",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.subdivide_phrases,
            default_value=False,
            on_select=on_select,
            subheading=SUBDIVIDE_SUBHEADING,
            hint=HINT_OUTE_LOUD_NORM if Tts.get_type() == TtsModelInfos.OUTE else None
        )

    @staticmethod
    def section_break_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.use_break_sound_effect = value
            state.project.save()
            print_feedback(f"Set to:", value)

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

        MAX = 35 # TODO parameterize menu-max-items
        file_infos = ProjectUtil.get_latest_concat_files(state.project, MAX)
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

def ask_output_indices_and_make(state: State) -> None:

    num_generated = state.project.sound_segments.num_generated()
    if not state.prefs.project_dir or num_generated == 0:
        print_feedback("Requires generated audio", is_error=True)
        return

    type_string = "AAC/M4B" if state.project.export_type == ExportType.AAC else "FLAC"

    should_ask_file_numbers = (state.project.chapter_mode == SectionMarkerMode.FILES) and len(state.project.markers) > 0
    if should_ask_file_numbers:
        
        infos = OutputRangeInfo.make_output_range_infos(state.project)

        result = ask_output_indices(infos)
        if result is None:
            return
        else:
            output_indices = result
            bookmark_indices = []

        noun = make_noun('file', 'files', len(output_indices))
        s = f"Will create the following {type_string} {noun}:"
        printt(s)

        strings = make_output_range_info_strings(infos, output_indices)
        s = "\n".join( ("    " + item) for item in strings)
        printt(s)
        printt()

        b = ask.ask_confirm()
        if not b:
            return

    else:

        output_indices = []
        if state.project.can_use_bookmark_section_markers():
            bookmark_indices = state.project.markers
        else:
            bookmark_indices = []

        info = OutputRangeInfo.make_single_info(state.project)
        s = f"Will create a single {type_string} file"
        printt(s)
        s = f"{COL_DIM}All lines "
        s += f"({info.num_files_exist}/{info.num_segments} generated){COL_DEFAULT}"
        printt(s)
        printt()

        b = ask.ask_confirm()
        if not b:
            return

    ConcatUtil.make_files(
        state=state, 
        file_cut_indices=output_indices, 
        bookmark_indices=bookmark_indices
    )

def ask_output_indices(infos: list[OutputRangeInfo]) -> list[int] | None:

    printt("Enter file numbers to create:")
    printt(f"{COL_DIM}(For example: \"1, 2, 4\" or  \"2-5\", or \"all\")")
    inp = ask.ask()

    if inp == "all" or inp == "a":
        indices = [info.output_index for info in infos if info.num_files_exist > 0]
        if not indices:
            print_feedback("No chapter files have generated audio", is_error=True)
            return None
        return indices

    indices, warnings = RangeStringUtil.parse_ranges_string(inp, len(infos))
    if warnings:
        message = "\n".join(warnings)
        print_feedback(message, is_error=True)
        return None

    indices = list(indices)
    if not indices:
        return None

    missing_audio_indices = [index for index in indices if infos[index].num_files_exist == 0]
    if missing_audio_indices:
        item_numbers = ", ".join(str(index + 1) for index in missing_audio_indices)
        print_feedback(f"No generated audio for chapter file: {item_numbers}", is_error=True)
        return None
    
    return indices

chromium_info = get_chromium_info()

LOUDNORM_SUBHEADING = \
"""Performs an extra pass after concatenation to minimize volume disparities between
TTS generations. The \"Stronger\" profile is a more aggressive setting, suitable 
for mobile devices.
"""

SUBDIVIDE_SUBHEADING = \
"""Affects how text is highlighted in the player/reader app.
When False, text segments map directly to the TTS prompts used to generate the audio segments.
When True, text is further sub-segmented by phrase  (Requires \"speech-to-text validation\" 
to be enabled during TTS sound generation).
"""

SECTION_BREAK_SUBHEADING = \
"""In the concatenation step, inserts a 'page turn' sound effect when 
two or more consecutive blank lines are encountered in the text. 
This can be a useful audible cue, so long as the text is formatted for it.
"""

OPEN_AUDIOBOOK_SUBHEADING = \
f"""Select file to be opened in the player app using %1, which will be 
launched with a dedicated user profile and the following flags 
to enable opening local audio files without user input:
  {COL_DIM}--allow-file-access-from-files 
  --autoplay-policy=no-user-gesture-required
  --user-data-dir=%2
"""

HIGH_SHELF_SUBHEADING = \
"""Applies a high-shelf equalizer pass to compensate for dull or muffled-sounding TTS output.
Some TTS models may benefit more from this than others.

This setting also applies to: 
Realtime playback, LLM voice chat, and stand-alone server
"""

UPSAMPLE_SUBHEADING = \
"""Uses Sidon generative model to upsample audio to 48kHz.
Enhances audio quality and clarity; affects timbre and tonality.
"""
