import os

from tts_audiobook_tool.app_types import ChapterMode, ExportType, NormalizationType
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.chapter_dividers_menu import ChapterDividersMenu
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_util import ProjectUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class ConcatMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_chapter_dividers_label(_: State) -> str:
            qty = len(state.project.section_dividers)
            label = "Chapter dividers "
            if qty > 0:
                noun = make_noun('item', 'items', qty)
                mode = state.project.chapter_mode.label.lower()
                value = f"{qty} {noun}, {mode}"
                label += make_currently_string(value)
            else:
                label += f"{COL_DIM}(optional)"
            return label

        def on_start(_: State, __: MenuItem) -> None:
            ask_chapter_indices_and_make(state)

        def make_items(_: State) -> list[MenuItem]:
            items = [
                MenuItem("Start", on_start),
                MenuItem(make_chapter_dividers_label, lambda _, __: ChapterDividersMenu.menu(state)),
                MenuItem(
                    lambda _: make_menu_label("File type", state.project.export_type.label), 
                    lambda _, __: ConcatMenu.file_type_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label("Loudness normalization", state.project.normalization_type.value.label),
                    lambda _, __: ConcatMenu.normalization_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label("Subdivide into phrases", state.project.subdivide_phrases), 
                    lambda _, __: ConcatMenu.subdivide_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label("Section break sound effect", state.project.use_section_sound_effect),
                    lambda _, __: ConcatMenu.section_break_menu(state)
                )
            ]

            can_and_haz = chromium_info and ProjectUtil.get_latest_concat_files(state.project)
            if can_and_haz:
                items.append(
                    MenuItem(
                        f"Open audiobook file in the player app", 
                        lambda _, __: ConcatMenu.open_audiobook_menu(state)
                    )
                )
            return items

        MenuUtil.menu(state, "Concatenate audio segments:", make_items, subheading=make_chapter_files_subheading)

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
    def subdivide_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.subdivide_phrases = value
            state.project.save()
            print_feedback(f"Subdivide into phrases set to: {state.project.subdivide_phrases}")

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
            state.project.use_section_sound_effect = value
            state.project.save()
            print_feedback(f"Set to:", value)

        MenuUtil.options_menu(
            state=state,
            heading_text="Section break sound effect",
            subheading=SECTION_BREAK_SUBHEADING,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.use_section_sound_effect,
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def open_audiobook_menu(state: State) -> None:
        
        if chromium_info is None:
            return

        MAX = 35 # TODO parameterize menu-max-items
        file_paths = ProjectUtil.get_latest_concat_files(state.project, MAX)
        if not file_paths:
            print_feedback("No files found")
            return
        
        user_data_dir=AppUtil.get_chromium_user_data_dir()

        def on_item(_: State, item: MenuItem) -> None:
            assert(isinstance(chromium_info, tuple))
            launch_player_with_chromium(
                chromium_path=chromium_info[1],
                audio_file_path=item.data,
                user_data_dir=user_data_dir
            )

        def make_items(_: State) -> list[MenuItem]:
            items = []
            for file_path in file_paths:
                label = Path(file_path).name
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

# ---

def make_chapter_files_subheading(state: State) -> str:

    if state.project.chapter_mode != ChapterMode.FILES:
        return ""

    infos = ChapterInfo.make_chapter_infos(state.project)
    if len(infos) == 1:
        return "" # bc always has one item
    
    if len(infos) > 4:
        subinfos = infos[:3]
        extra_line = f"Plus {len(infos) - len(subinfos)} more items"
    else:
        subinfos = infos
        extra_line = ""

    strings = make_chapter_info_strings( subinfos, list(range(len(subinfos))) )
    string = "\n".join(strings)
    string += "\n"
    if extra_line:
        string += extra_line + "\n"
    return string

def make_chapter_info_strings(infos: list[ChapterInfo], indices: list[int]) -> list[str]:
    lst = []
    for index in indices:
        s = make_chapter_files_string(infos[index], index)
        lst.append(s)
    return lst

def make_chapter_files_string(info: ChapterInfo, index: int) -> str:
    s = f"{COL_DEFAULT}Chapter file {index+1}:{COL_DIM} lines {info.segment_index_start + 1} to {info.segment_index_end + 1} "
    missing = f", {info.num_files_missing} missing" if info.num_files_missing else ""
    s += f"({info.num_files_exist}/{info.num_segments} generated{missing}){COL_DEFAULT}"
    return s

def ask_chapter_indices_and_make(state: State) -> None:

    type_string = "AAC/M4B" if state.project.export_type == ExportType.AAC else "FLAC"

    should_ask_file_numbers = (state.project.chapter_mode == ChapterMode.FILES) and len(state.project.section_dividers) > 0
    if should_ask_file_numbers:
        
        infos = ChapterInfo.make_chapter_infos(state.project)

        result = ask_chapter_indices(infos, len(state.project.phrase_groups))
        if result is None:
            return
        else:
            chapter_indices = result
            bookmark_indices = []

        noun = make_noun('file', 'files', len(chapter_indices))
        s = f"Will create the following {type_string} {noun}:"
        printt(s)

        strings = make_chapter_info_strings(infos, chapter_indices)
        s = "\n".join( ("    " + item) for item in strings)
        printt(s)
        printt()

        b = AskUtil.ask_confirm()
        if not b:
            return

    else:

        chapter_indices = []
        bookmark_indices = state.project.section_dividers

        info = ChapterInfo.make_single_info(state.project)
        s = f"Will create a single {type_string} file"
        printt(s)
        s = f"{COL_DIM}All lines "
        s += f"({info.num_files_exist}/{info.num_segments} generated){COL_DEFAULT}"
        printt(s)
        printt()

        b = AskUtil.ask_confirm()
        if not b:
            return

    ConcatUtil.make_files(
        state=state, 
        chapter_indices=chapter_indices, 
        bookmark_indices=bookmark_indices
    )

def ask_chapter_indices(infos: list[ChapterInfo], num_phrase_groups: int) -> list[int] | None:

    # Filter out items w/o any generated files
    infos = [info for info in infos if info.num_files_exist > 0] 

    printt("Enter chapter file numbers to create:")
    printt(f"{COL_DIM}(For example: \"1, 2, 4\" or  \"2-5\", or \"all\")")
    inp = AskUtil.ask()

    if inp == "all" or inp == "a":
        indices = [info.chapter_index for info in infos]
        return indices

    indices, warnings = ParseUtil.parse_ranges_string(inp, num_phrase_groups)
    if warnings:
        message = "\n".join(warnings)
        print_feedback(message, is_error=True)
        return None

    indices = list(indices)
    if not indices:
        return None

    valid_indices = {info.chapter_index for info in infos}
    for index in indices:
        if index not in valid_indices:
            print_feedback(f"Item out of range: {index + 1}", is_error=True)
            return None
    
    return indices

def launch_player_with_chromium(
        chromium_path: str, 
        audio_file_path: str,
        user_data_dir: str
    ) -> None:
    """
    Launches local player/reader in Chrome/Chromium with the latest concat'ed file

    Eg:
      chromium
          --allow-file-access-from-files
          --autoplay-policy=no-user-gesture-required
          --user-data-dir=/path/to/tts-audiobook-tool-chromium-profile
          file:///path/to/index.html?url=/path/to/audiobook.m4b
    """

    if user_data_dir:
        os.makedirs(user_data_dir, exist_ok=True)
        if not os.path.exists(user_data_dir):
            print_feedback(f"Couldn't create browser user directory {user_data_dir}")
            return

    browser_flags = [
        "--allow-file-access-from-files", 
        "--autoplay-policy=no-user-gesture-required",
    ]
    if user_data_dir:
        browser_flags.append(f"--user-data-dir={user_data_dir}")

    this_file_path = Path(os.path.abspath(__file__))
    index_html_path = this_file_path.parent.parent / "browser_player" / "index.html"
    index_html_url = index_html_path.as_uri() # important
    browser_url = make_url_with_params(index_html_url, { "url": audio_file_path })

    command = []
    command.append(chromium_path)
    command.extend(browser_flags)
    command.append(browser_url)

    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from Python process so it doesn't block
        )
        print_feedback(f"Launched process:\n{command}")
    
    except (FileNotFoundError, Exception) as e:
        print_feedback(make_error_string(e), is_error=True)


chromium_info = AppUtil.get_chromium_info()

LOUDNORM_SUBHEADING = \
"""Performs an extra pass after concatenating audio segments to minimize volume disparities
between TTS generations. The \"Stronger\" profile is a more aggressive setting, suitable 
for mobile devices.
"""

SUBDIVIDE_SUBHEADING = \
"""Affects how text is highlighted in the player/reader app.
When False, text segments map directly to the TTS prompts used to generate the audio segments.
When True, text is further sub-segmented by phrase. Requires \"speech-to-text validation\"
to be enabled during TTS sound generation.
"""

SECTION_BREAK_SUBHEADING = \
"""In the concatenation step, inserts a 'page turn' sound effect when 
two or more consecutive blank lines are encountered in the text. 
This can be a useful audible cue, so long as the text is formatted for it.
"""

OPEN_AUDIOBOOK_SUBHEADING = \
f"""Select file to be opened in the player app using %1,
which will be launched with a dedicated user profile and the 
following flags to enable opening local files without user input:
  {COL_DIM}--allow-file-access-from-files 
  --autoplay-policy=no-user-gesture-required
  --user-data-dir=%2
"""
