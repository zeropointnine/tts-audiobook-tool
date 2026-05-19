from __future__ import annotations

from pathlib import Path
import time

from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.constants import *


def show_hint_if_necessary(prefs, hint: Hint, and_confirm: bool=False, and_prompt: bool=False) -> bool:
    """
    Shows hint only if not yet shown.
    """
    from tts_audiobook_tool.prefs import Prefs

    assert(isinstance(prefs, Prefs))
    if prefs.get_hint(hint.key):
        return True
    prefs.set_hint_true(hint.key)
    return show_hint(hint, and_confirm=and_confirm, and_prompt=and_prompt)


def show_hint(hint: Hint, and_confirm: bool=False, and_prompt: bool=False) -> bool:
    """
    Shows hint.
    Then either asks for confirmation, prompts to press enter, or or shows a 3-second 'animation'
    Returns True if "should continue"
    """
    from tts_audiobook_tool import ask

    print_hint(hint)

    if and_confirm:
        return ask.ask_confirm()
    elif and_prompt:
        ask.ask_enter_to_continue()
        return True
    else:
        # Anim
        lines = ["[   ]", "[.  ]", "[.. ]", "[...]"]
        for line in lines:
            print(f"{COL_DIM}{line}{Ansi.RESET}", end="\r", flush=True)
            time.sleep(0.5)
        print(f"{Ansi.ERASE_REST_OF_LINE}", end="", flush=True)
        return True


def print_hint(hint: Hint) -> None:
    from tts_audiobook_tool.util import printt

    printt(f"🔔 {COL_ACCENT}{hint.heading}")
    printt(hint.text)
    printt()


def show_player_hint_if_necessary(prefs) -> None:
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.util import get_package_dir

    assert(isinstance(prefs, Prefs))

    s = "You can open audio files with the interactive player/reader here:\n"
    package_dir = get_package_dir()
    if package_dir:
        browser_path = str(Path(package_dir).parent / "browser_player" / "index.html")
    else:
        browser_path = "browser_player" + os.path.sep + "index.html"
    s += browser_path + "\n"
    s += "or on the web here:" + "\n"
    s += PLAYER_URL

    hint = Hint(key="player", heading="Reminder", text=s)
    show_hint_if_necessary(prefs, hint)