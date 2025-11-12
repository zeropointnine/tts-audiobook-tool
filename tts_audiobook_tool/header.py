import sys

from tts_audiobook_tool.constants_config import *


class Header:
    """
    Prints persistent header on top two lines of console display

    Does nothing if `HEADER_FEATURE` is False
    """

    @staticmethod
    def init():

        if not HEADER_FEATURE:
            return

        # Clear screen
        print("\033[2J", end="")
        # Set scroll region (lines 4 to bottom, leaving top 3 for header + blank line)
        print("\033[4;r", end="")
        # Move cursor to line 4 (start of scroll region)
        print("\033[4;1H", end="")
        sys.stdout.flush()

    @staticmethod
    def cleanup_console():

        if not HEADER_FEATURE:
            return

        """Restore normal terminal behavior"""
        # Reset scroll region to full screen
        print("\033[r", end="")
        # Move cursor to bottom
        print("\033[999;1H", end="")
        sys.stdout.flush()


    @staticmethod
    def update(line1, line2):
        """Update the two header lines without affecting scroll position"""

        if not HEADER_FEATURE:
            return

        # TODO: prevent word wrap?

        # Save current cursor position
        print(SAVE_CURSOR, end="")

        # Move to line 1, clear it, and write first header line
        print("\033[1;1H", end="")
        print(CLEAR_LINE, end="")
        print(line1, end="")

        # Move to line 2, clear it, and write second header line
        print("\033[2;1H", end="")
        print(CLEAR_LINE, end="")
        print(line2, end="")

        # Restore cursor position
        print(RESTORE_CURSOR, end="")
        sys.stdout.flush()

# ---

SAVE_CURSOR = "\033[s"
RESTORE_CURSOR = "\033[u"
CLEAR_LINE = "\033[2K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
