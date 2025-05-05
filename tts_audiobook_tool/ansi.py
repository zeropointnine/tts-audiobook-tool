import os
import math

# Standard 256-color palette (indices 16-231 are a 6x6x6 color cube)
# Standard 256-color palette (indices 232-255 are grayscale)
# We'll use these for approximation

def _rgb_to_xterm256(r, g, b):
    """ Converts RGB values to the nearest xterm 256 color index. """
    # Check simple grayscale first
    if abs(r - g) < 8 and abs(g - b) < 8: # Allow slight deviation for grayscale
        gray = round((r + g + b) / 3)
        if gray < 8:
            return 16 # Black
        if gray > 248:
            return 231 # Use cube white (often better than grayscale 255)
        # Map 8-248 to 232-255 (24 steps)
        gray_index = round(((gray - 8) / 240) * 23) # 248-8 = 240 range
        return 232 + gray_index

    # If not grayscale, map to the 6x6x6 color cube (indices 16-231)
    # Convert 0-255 range to 0-5 range for cube indices
    r_idx = round((r / 255) * 5)
    g_idx = round((g / 255) * 5)
    b_idx = round((b / 255) * 5)
    return 16 + (r_idx * 36) + (g_idx * 6) + b_idx

class Ansi:
    """ Low level print strings, mostly ANSI """

    RESET = "\033[0m"

    CLEAR_SCREEN_HOME: str = "\033[2J\033[H"
    SCREEN_HOME: str = "\x1b[1;1H"
    LINE_HOME: str = "\x1b[1G"

    ERASE_REST_OF_LINE: str = "\033[K"

    CURSOR_HIDE: str = "\033[?25l"
    CURSOR_SHOW: str = "\033[?25h"

    ITALICS: str = "\x1b[3m"
    STRIKETHROUGH: str = "\x1b[9m"

    @staticmethod
    def hex(hex_color: str, is_background=False) -> str:
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        # Ensure hex_color is exactly 6 characters long, padding if necessary
        hex_color = hex_color.ljust(6, '0')[:6]
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except ValueError:
            # Handle invalid hex input gracefully, e.g., return reset or default color
            r, g, b = 255, 255, 255 # Default to white on error

        colorterm = os.environ.get('COLORTERM', '').lower()
        supports_truecolor = colorterm in ('truecolor', '24bit')

        base = 48 if is_background else 38  # 48 for background, 38 for foreground

        if supports_truecolor:
            # Use 24-bit true color
            return f"\033[{base};2;{r};{g};{b}m"
        else:
            # Fallback to 256-color approximation
            color_index = _rgb_to_xterm256(r, g, b)
            return f"\033[{base};5;{color_index}m"

    @staticmethod
    def cursor_pos(row: int, col: int) -> str:
        """
        Values are 1-indexed
        """
        # Basic validation for row/col
        row = max(1, int(row))
        col = max(1, int(col))
        return f"\033[{row};{col}H"