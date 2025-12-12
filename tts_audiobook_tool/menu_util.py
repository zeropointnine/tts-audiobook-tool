from __future__ import annotations
import inspect
from typing import Callable, cast

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from typing import TypeVar, Callable, Any

T = TypeVar("T")

class MenuItem:
    def __init__(
            self,
            label: StringOrMaker,
            handler: MenuHandler,
            data: Any = None,
            sublabel: StringOrMaker | None = None,
            hotkey: str = ""
    ):
        self.label = label

        # Optional extra text printed on second line
        self.sublabel = sublabel

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

    @staticmethod
    def menu(
        state: State,
        heading: StringOrMaker,
        items: MenuItemListOrMaker,
        is_submenu: bool = True,
        subheading: StringOrMaker | None = None,
        hint: Hint | None = None,
        one_shot: bool = False,
        on_exit: Callable | None = None
    ):
        """
        Prints a menu of items, waits for input, and executes mapped callback.

        When readchar is enabled, it blocks until recognized hotkey is pressed.
        Otherwise, it repeats until unrecognized key is entered or enter is pressed

        param one_shot:
            If True, exits after executing mapped callback function 
            (ie, in practice, goes to previous menu)

        param on_exit
            Gets called when menu exits (ie, when app goes *back* to previous menu)
        """

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
                # Assign hotkeys "1", "2", "3", etc
                for i, item in enumerate(items_list):
                    item.hotkey = str(i + 1)
            else:
                # Verify no dupes
                hotkeys = set[str]()
                for item in items_list:
                    if item.hotkey in hotkeys:
                        raise ValueError(f"Duplicate hotkey {item.hotkey}")
                    hotkeys.add(item.hotkey)

            # Print heading
            s = get_string_from(state, heading)
            print_heading(s)

            # Print optional subheading
            if subheading:
                s = get_string_from(state, subheading)
                if s:
                    printt(s)

            # Print optional hint
            if hint:
                AppUtil.show_hint_if_necessary(state.prefs, hint)

            # Print items
            for item in items_list:
                s = get_string_from(state, item.label)
                s = make_hotkey_string(item.hotkey.upper()) + " " + s
                if item.sublabel:
                    # Print extra line/s
                    sublabel = get_string_from(state, item.sublabel)
                    space = "    " if not sublabel.startswith(" ") else ""
                    s += "\n" + COL_DIM + space + sublabel
                printt(s)
            printt()

            # One-time message
            if is_submenu and MenuUtil.is_first_submenu:
                MenuUtil.is_first_submenu = False
                printt(f"{COL_DIM}Press {COL_DEFAULT}{make_hotkey_string('Enter')}{COL_DIM} to go back one level")
                printt()

            while True:
                # Prompt
                hotkey = AskUtil.ask_hotkey()

                if AskUtil.is_readchar:
                    # enter (windows), enter (mac), backspace, escape
                    should_return = hotkey in ["\r", "\n", "\x08", "\x1b"] and is_submenu
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
                    if AskUtil.is_readchar:
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
        subheading: str=""
    ) -> None:
        """
        Displays a menu with a list of values.
        Think radio button group.
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
            one_shot=True
        )

# ---

def get_string_from(state: State, string_or_maker: StringOrMaker) -> str:
    if isinstance(string_or_maker, str):
        return string_or_maker
    else:
        return string_or_maker(state)