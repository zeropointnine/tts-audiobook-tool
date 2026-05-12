from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Callable
from tts_audiobook_tool.app_types import SttVariant
import tts_audiobook_tool.util as util_module

from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from typing import TypeVar, Callable, Any

T = TypeVar("T")


@dataclass
class MenuFrame:
    heading: StringOrMaker
    breadcrumb: StringOrMaker | None = None

class MenuItem:
    def __init__(
            self,
            label: StringOrMaker,
            handler: MenuHandler,
            data: Any = None,
            sublabel: StringOrMaker | None = None,
            hotkey: str = "",
            superlabel: StringOrMaker = "",
            superlabel_no_blank_line: bool = False
    ):
        self.label = label

        # Optional extra text printed on second line
        self.sublabel = sublabel

        # Optional label printed on its own line above the item
        self.superlabel = superlabel
        self.superlabel_no_blank_line = superlabel_no_blank_line

        # handler/callback passes the State object and `data`, if any
        self.handler = handler

        # Optional callback data
        self.data = data

        # The hotkey that will trigger the handler
        self.hotkey = hotkey

# ---

# Type aliases:

# A function that returns a string
StringMaker = Callable[[State], str]

# A string or a function that returns a string
StringOrMaker = StringMaker | str

# A list of MenuItems or a function that returns a list of MenuItems
MenuItemListOrMaker = Callable[[State], list[MenuItem]] | list[MenuItem]

# Callback function for when an item is selected; returns True if menu should then exit
MenuHandler = Callable[[State, MenuItem], bool | None]

# ---

class MenuUtil:

    is_first_submenu = True
    menu_frames: list[MenuFrame] = []

    @staticmethod
    def menu(
        state: State,
        heading: StringOrMaker,
        items: MenuItemListOrMaker,
        is_submenu: bool = True,
        subheading: StringOrMaker | None = None,
        hint: Hint | None = None,
        one_shot: bool = False,
        on_exit: Callable | None = None,
        breadcrumb: StringOrMaker | None = None,
    ):
        """
        Prints a menu of items, waits for input, and executes mapped callback.

        When single-key hotkey input is enabled, it blocks until recognized hotkey is pressed.
        Otherwise, it repeats until unrecognized key is entered or enter is pressed

        param one_shot:
            If True, exits after executing mapped callback function 
            (ie, in practice, goes to previous menu)

        param on_exit
            Gets called when menu exits (ie, when app goes *back* to previous menu)
        """

        MenuUtil.menu_frames.append(MenuFrame(heading=heading, breadcrumb=breadcrumb))
        try:
            while True:

                # Initialize items list
                if isinstance(items, list):
                    items_list: list[MenuItem] = items
                else:
                    items_list = items(state)

                is_all_hotkeys = all(item.hotkey for item in items_list)
                is_no_hotkeys = all(not item.hotkey for item in items_list)

                if not is_all_hotkeys and not is_no_hotkeys:
                    raise ValueError("All MenuItems must have hotkeys or no MenuItems must have hotkeys")

                if is_no_hotkeys:
                    # Assign hotkeys to items
                    HOTKEYS_AUTO = list('123456789abcdefghijklmnopqrstuvwxyz')
                    items_list = items_list[:len(HOTKEYS_AUTO)] # Silently clamp
                    for i, item in enumerate(items_list):
                        item.hotkey = HOTKEYS_AUTO[i]
                else:
                    # Verify no dupes
                    hotkeys = set[str]()
                    for item in items_list:
                        if item.hotkey in hotkeys:
                            raise ValueError(f"Duplicate hotkey {item.hotkey}")
                        hotkeys.add(item.hotkey)

                # Print heading
                s = get_string_from(state, heading)
                breadcrumb_text = MenuUtil.make_breadcrumb_text(state)
                MenuUtil.print_heading(state, s, breadcrumb_text=breadcrumb_text)

                # Print optional subheading
                if subheading:
                    s = get_string_from(state, subheading)
                    if s:
                        printt(s)

                # Print optional hint
                if hint:
                    Hint.show_hint_if_necessary(state.prefs, hint)

                left_padding = "  "

                # Print items
                for item in items_list:
                    if item.superlabel:
                        superlabel_text = get_string_from(state, item.superlabel)
                        if superlabel_text:
                            if not item.superlabel_no_blank_line:
                                printt()
                            printt(f"{left_padding}{COL_DIM}{superlabel_text}")
                    s = get_string_from(state, item.label)
                    s = left_padding + make_hotkey_string(item.hotkey.upper()) + " " + s
                    if item.sublabel:
                        # Print extra line/s
                        sublabel = get_string_from(state, item.sublabel)
                        space = ("    " + left_padding) if not sublabel.startswith(" ") else left_padding
                        s += "\n" + COL_DIM + space + sublabel
                    printt(s)
                printt()

                # One-time message
                if is_submenu and MenuUtil.is_first_submenu:
                    MenuUtil.is_first_submenu = False
                    printt(
                        f"{left_padding}{COL_DIM}Press {COL_DEFAULT}{make_hotkey_string('Enter')}{COL_DIM} or "
                        f"{COL_DEFAULT}{make_hotkey_string('Esc')}{COL_DIM} to go back one level"
                    )
                    printt()

                while True:
                    # Prompt
                    hotkey = AskUtil.ask_hotkey()

                    if AskUtil.can_hotkey:
                        # enter (windows), enter (mac/linux), backspace, escape
                        # Some terminals/readchar combos can return double-escape for Esc.
                        should_return = hotkey in ["\r", "\n", "\x08", "\x1b", "\x1b\x1b"] and is_submenu
                    else: # can't readchar
                        should_return = not hotkey
                    if should_return:
                        if on_exit:
                            on_exit()
                        return

                    # Handle hotkey
                    selected_item = None
                    for item in items_list:
                        if item.hotkey == hotkey:
                            selected_item = item
                            break

                    if selected_item:
                        break
                    else:
                        if AskUtil.can_hotkey:
                            continue # wait for next hotkey
                        else:
                            return

                # Execute mapped callback function
                should_return = selected_item.handler(state, selected_item)
                if should_return is True:
                    if on_exit:
                        on_exit()
                    return

                if one_shot:
                    if on_exit:
                        on_exit()
                    return
        finally:
            MenuUtil.menu_frames.pop()

    @staticmethod
    def make_breadcrumb_text(state: State) -> str:
        if not state.prefs.menu_clears_screen:
            return ""

        ancestor_frames = MenuUtil.menu_frames[:-1]
        if not ancestor_frames:
            return ""

        segments = [
            segment
            for segment in [MenuUtil.make_breadcrumb_segment(state, frame) for frame in ancestor_frames]
            if segment
        ]
        if not segments:
            return ""

        return " > ".join(segments) + " >"

    @staticmethod
    def make_breadcrumb_segment(state: State, frame: MenuFrame) -> str:
        string_or_maker = frame.breadcrumb if frame.breadcrumb is not None else frame.heading
        segment = get_string_from(state, string_or_maker)
        segment = strip_ansi_codes(segment).strip().rstrip(":")

        if frame.breadcrumb is None:
            segment = re.sub(r"\s*\([^)]*\)\s*$", "", segment).strip()

        return segment

    @staticmethod
    def print_screen_heading(
        state: State,
        heading: StringOrMaker,
        breadcrumb: StringOrMaker | None = None,
    ) -> None:
        """
        Prints a heading for a blocking prompt/screen that is not a full menu,
        but should still participate in the current breadcrumb trail.

        This temporarily pushes a MenuFrame so existing ancestor-only
        breadcrumb rendering works the same way as MenuUtil.menu(...), while
        keeping direct print_heading(...) calls breadcrumb-free by default.
        """
        MenuUtil.menu_frames.append(MenuFrame(heading=heading, breadcrumb=breadcrumb))
        try:
            heading_text = get_string_from(state, heading)
            breadcrumb_text = MenuUtil.make_breadcrumb_text(state)
            MenuUtil.print_heading(state, heading_text, breadcrumb_text=breadcrumb_text)
        finally:
            MenuUtil.menu_frames.pop()

    @staticmethod
    def print_heading(
        state: State | None,
        text: str,
        dont_clear: bool=False,
        non_menu: bool=False,
        breadcrumb_text: str="",
    ) -> None:
        """
        :param dont_clear: Doesn't clear screen even when settings are "menu clears screen"

        TODO: Params are a mess at this point
        """

        if state and state.prefs.menu_clears_screen and not dont_clear and not non_menu:
            os.system('cls' if os.name == 'nt' else 'clear')
            MenuUtil._print_status_block(state)
            printt()
        elif util_module._menu_clears_screen and not dont_clear:
            os.system('cls' if os.name == 'nt' else 'clear')

        if state and not state.prefs.menu_clears_screen:
            length = len(strip_ansi_codes(text))
            print("-" * length)

        color = COL_DEFAULT if non_menu else COL_ACCENT
        if breadcrumb_text:
            printt(f"{COL_DIM}{breadcrumb_text}")
        printt(f"{color}{text}")
        printt()

    @staticmethod
    def _print_status_block(state: State) -> None:
        
        from tts_audiobook_tool.app_util import AppUtil
        from tts_audiobook_tool.stt import Stt
        from tts_audiobook_tool.tts import Tts

        label_color = COL_DIM
        value_color = COL_MEDIUM

        if state.project.dir_path:
            path_string = value_color + make_terminal_hyperlink(state.project.dir_path)
        else:
            path_string = value_color + "none"

        language_code = state.project.language_code.strip()
        if state.project.dir_path and language_code:
            path_string += f" {COL_DIM}({language_code})"

        if Tts._instance_display_info:
            tts_model_text = value_color + Tts._instance_display_info.model_description
            match (bool(Tts._instance_display_info.device), bool(Tts._instance_display_info.extra)):
                case (True, True):
                    s = f"{Tts._instance_display_info.device}, {Tts._instance_display_info.extra}"
                case (True, False):
                    s = Tts._instance_display_info.device
                case (False, True):
                    s = Tts._instance_display_info.extra
                case (False, False):
                    s = ""
            if s:
                tts_model_text += f" {COL_DIM}({s})"
            tts_loaded = "(loaded)"
        else:
            tts_model_text = value_color + Tts.get_class().INFO.ui['proper_name']
            tts_loaded = "(not loaded)"

        voice_prefix, voice_value = Tts.get_class().get_voice_display_info(
            state.project,
            Tts.get_instance_if_exists()
        )
        voice_prefix = strip_ansi_codes(voice_prefix).strip().rstrip(":")
        voice_value = strip_ansi_codes(voice_value).strip()
        voice_text = value_color + (voice_value or voice_prefix or "none")

        total_lines = len(state.project.phrase_groups)
        num_complete = state.project.sound_segments.num_generated()
        text_text = value_color + f"{total_lines} lines"
        text_text += f" {COL_DIM}({num_complete} segments generated)"

        stt_model = "mlx-whisper" if Stt.should_use_mlx_whisper() else "faster-whisper"
        stt_desc = value_color + stt_model
        if state.prefs.stt_variant == SttVariant.DISABLED:
            stt_desc += f" disabled" # not dim
        else:
            if Stt.has_instance():
                stt_variant = Stt.get_variant().id
                fw_config = state.prefs.stt_config.description if not Stt.should_use_mlx_whisper() else ""
                stt_desc += f" {stt_variant} {COL_DIM}({fw_config}) {COL_DIM}(loaded)"            
            else:
                stt_desc += f" {COL_DIM}(not loaded)"

        memory_text = strip_ansi_codes(AppUtil.make_memory_string())
        memory_text = memory_text.replace(":", "") # careful
        memory_text = value_color + memory_text if memory_text else ""

        printt(f"{label_color}Project:     {path_string}")
        printt(f"{label_color}TTS model:   {tts_model_text} {COL_DIM}{tts_loaded}")
        printt(f"{label_color}Voice clone: {voice_text}")
        printt(f"{label_color}Text:        {text_text}")
        printt(f"{label_color}STT model:   {stt_desc}")
        if memory_text:
            printt(f"{label_color}Memory:      {memory_text}")


    @staticmethod
    def options_menu(
        state: State,
        heading_text: str,
        labels: list[str],
        values: list[T],
        current_value: T | None,
        default_value: T | None,
        on_select: Callable[[T], None],
        sublabels: list[str] | None = None,
        hint: Hint | None=None,
        subheading: str="",
        breadcrumb: StringOrMaker | None = None,
    ) -> None:
        """
        Displays a menu with a list of values.
        Think drop-down-list or radio button group.
        If an item is selected, calls `on_select` (returns string+value tuple)

        `labels`, `values`, and `sublabels` (if exists) are all "parallel lists"
        """
        if len(labels) != len(values):
            raise ValueError("labels and values lists must have same size")
        if sublabels and len(sublabels) != len(labels):
            raise ValueError("labels and sublabels lists must have same size")

        def on_menu_item(_: State, item: MenuItem) -> None:
            on_select(item.data) # callback

        items: list[MenuItem] = []
        for i in range(0, len(labels)):
            label = labels[i]
            value = values[i]
            sublabel = sublabels[i] if sublabels else None
            if value == default_value:
                label += f" {COL_DIM}(default)"
            if value == current_value:
                label += f" {COL_ACCENT}(selected)"
            menu_item = MenuItem(label, on_menu_item, value, sublabel=sublabel)
            items.append(menu_item)

        MenuUtil.menu(
            state=state,
            heading=heading_text,
            items=items,
            hint=hint,
            subheading=subheading,
            one_shot=True,
            breadcrumb=breadcrumb,
        )

    @staticmethod
    def make_number_label(
        project: Project,
        attr: str,
        base_label: str,
        default_value: int | float | None = None,
        is_minus_one_default: bool = True,
        num_decimals: int = 1,
    ) -> str:

        label_value: float | int | None = getattr(project, attr, None)
        if label_value is None:
            raise ValueError(f"Attribute doesn't exist: {attr}")
        if is_minus_one_default and label_value == -1:
            if default_value is None:
                raise ValueError("Default value required")
            label_value = default_value
        label = make_menu_label(base_label, label_value, default_value, num_decimals=num_decimals)
        return label

    @staticmethod
    def make_number_item(
        state: State, 
        attr: str,
        base_label: str,
        default_value: int | float,
        is_minus_one_default: bool,
        num_decimals: int,
        prompt: str,
        min_value: int | float,
        max_value: int | float
    ) -> MenuItem:
        """ 
        Makes "self-contained" MenuItem that displays a number value,
        and does stock "ask_number()" action on select.
        """

        if isinstance(min_value, int) and isinstance(max_value, int) and (isinstance(default_value, int) or default_value is None):
            is_int = True
        elif isinstance(min_value, float) and isinstance(max_value, float) and (isinstance(default_value, float) or default_value is None):
            is_int = False
        else:
            raise ValueError(f"Ambiguous argument types: {default_value}, {min_value}, {max_value}")
        
        if is_int and num_decimals > 0:
            num_decimals = 0

        def on_item(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                state.project,
                attr,
                prompt,
                min_value, max_value,
                default_value,
                "Value set:",
                is_int=is_int
            )

        label = MenuUtil.make_number_label(
            project=state.project,
            attr=attr,
            base_label=base_label,
            default_value=default_value,
            is_minus_one_default=is_minus_one_default,
            num_decimals=num_decimals
        )

        return MenuItem(label, on_item)

# ---

def should_show_menu_status_details(state: State) -> bool:
    return not state.prefs.menu_clears_screen

def get_string_from(state: State, string_or_maker: StringOrMaker) -> str:
    if isinstance(string_or_maker, str):
        return string_or_maker
    else:
        return string_or_maker(state)
    
HOTKEYS_AUTO = list('123456789abcdefghijklmnopqrstuvwxyz')