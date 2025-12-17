from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
from tts_audiobook_tool.constants import *

@dataclass
class Hint:
    key: str
    heading: str
    text: str

    @staticmethod
    def show_hint_if_necessary(prefs, hint: Hint, and_confirm: bool=False, and_prompt: bool=False) -> bool:
        """
        Shows hint only if not yet shown.
        """
        from tts_audiobook_tool.prefs import Prefs
        assert(isinstance(prefs, Prefs))
        if prefs.get_hint(hint.key):
            return True
        prefs.set_hint_true(hint.key)
        return Hint.show_hint(hint, and_confirm=and_confirm, and_prompt=and_prompt)

    @staticmethod
    def show_hint(hint: Hint, and_confirm: bool=False, and_prompt: bool=False) -> bool:
        """
        Shows hint.
        Then either asks for confirmation, prompts to press enter, or or shows a 3-second 'animation'
        Returns True if "should continue"
        """
        from tts_audiobook_tool.ask_util import AskUtil

        Hint.print_hint(hint)

        if and_confirm:
            return AskUtil.ask_confirm()
        elif and_prompt:
            AskUtil.ask_enter_to_continue()
            return True
        else:
            # Anim
            lines = ["[   ]", "[.  ]", "[.. ]", "[...]"]
            for i, line in enumerate(lines):
                print(f"{COL_DIM}{line}{Ansi.RESET}", end="\r", flush=True)
                time.sleep(0.66)
            print(f"{Ansi.ERASE_REST_OF_LINE}", end="", flush=True)
            return True

    @staticmethod
    def print_hint(hint: Hint) -> None:
        from tts_audiobook_tool.util import printt
        printt(f"🔔 {COL_ACCENT}{hint.heading}")
        printt(hint.text)
        printt()

    @staticmethod
    def show_player_hint_if_necessary(prefs) -> None:
        
        from tts_audiobook_tool.prefs import Prefs
        assert(isinstance(prefs, Prefs))
        from tts_audiobook_tool.util import get_package_dir
        
        s = "You can open audio files with the interactive player/reader here:\n"
        package_dir = get_package_dir()
        if package_dir:
            browser_path = str( Path(package_dir).parent / "browser_player" / "index.html" )
        else:
            browser_path = "browser_player" + os.path.sep + "index.html"
        s += browser_path + "\n"
        s += "or on the web here:" + "\n"
        s += PLAYER_URL

        hint = Hint(key="player", heading="Reminder", text = s)
        Hint.show_hint_if_necessary(prefs, hint)

