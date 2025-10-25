from __future__ import annotations
from typing import Callable, NamedTuple, Set

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class MenuItem:
    def __init__(
            self,
            label: StringOrMaker,
            handler: Callable[[State, str], Any],
            data: Any = None,
            hotkey: str = ""
    ):
        self.label = label

        # handler/callback passes the State object and `data`, if any
        self.handler = handler

        # Optional callback data
        self.data = data

        # The hotkey that will trigger the handler
        self.hotkey = hotkey


StringOrMaker = Callable[[State], str] | str
MenuItemListOrMaker = Callable[[State], list[MenuItem]] | list[MenuItem]


class MenuUtil:

    # TODO: where and how to trigger hint action

    is_first_submenu = True

    @staticmethod
    def menu(
        state: State,
        heading: StringOrMaker,
        menu_items: MenuItemListOrMaker,
        is_submenu: bool = True,
        hint: Hint | None = None,
        one_shot: bool = False
    ):
        """
        Prints a menu of items, waits for input, and executes mapped callback
        If one_shot is False, repeats until unrecognized key is entered or enter is pressed
        """

        while True:

            # Initialize items
            if isinstance(menu_items, list):
                items = menu_items
            else:
                items = menu_items(state)

            is_all_hotkeys = all(item.hotkey for item in items)
            is_no_hotkeys = all(not item.hotkey for item in items)

            if not is_all_hotkeys and not is_no_hotkeys:
                raise ValueError("All MenuItems must have hotkeys or no MenuItems must have hotkeys")

            if is_no_hotkeys:
                # Assign hotkeys "1", "2", "3", etc
                for i, item in enumerate(items):
                    item.hotkey = str(i + 1)
            else:
                # Verify no dupes
                hotkeys = set[str]()
                for item in items:
                    if item.hotkey in hotkeys:
                        raise ValueError(f"Duplicate hotkey {item.hotkey}")
                    hotkeys.add(item.hotkey)

            # Print heading
            if isinstance(heading, str):
                s = heading
            else:
                s = heading(state)
            print_heading(s)

            # Print hint if necessary
            if hint:
                AppUtil.show_hint_if_necessary(state.prefs, hint)

            # Print items
            for item in items:
                if isinstance(item.label, str):
                    s = item.label
                else:
                    s = item.label(state)
                s = make_hotkey_string(item.hotkey.upper()) + " " + s
                printt(s)
            printt()

            # One-time message
            if is_submenu and MenuUtil.is_first_submenu:
                MenuUtil.is_first_submenu = False
                printt(f"{COL_DIM}Press {COL_DEFAULT}{make_hotkey_string('Enter')}{COL_DIM} to go back one level")
                printt()

            # Prompt
            hotkey = ask_hotkey()
            if not hotkey:
                return

            # Handle hotkey
            selected_item = None
            for item in items:
                if item.hotkey == hotkey:
                    selected_item = item
                    break
            if not selected_item:
                return
            should_return = selected_item.handler(state, selected_item.data)
            if should_return is True:
                return

            if one_shot:
                return
