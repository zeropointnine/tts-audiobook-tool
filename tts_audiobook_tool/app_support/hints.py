from __future__ import annotations

import time

from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.prefs import Prefs


def show_hint_if_necessary(prefs: Prefs, hint: Hint, and_confirm: bool=False, and_prompt: bool=False) -> bool:
    """
    Shows hint only if not yet shown.
    Returns True if "should continue"
    """
    from tts_audiobook_tool.prefs import Prefs
    assert(isinstance(prefs, Prefs))
    if prefs.get_hint(hint.key):
        return True
    should_continue = show_hint(hint, and_confirm=and_confirm, and_prompt=and_prompt)
    if should_continue:
        prefs.set_hint_true(hint.key)
    return should_continue


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
