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
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        base = 48 if is_background else 38  # 48 for background, 38 for foreground
        return f"\033[{base};2;{r};{g};{b}m"

    @staticmethod
    def cursor_pos(row: int, col: int) -> str:
        """
        Values are 1-indexed
        """
        return f"\033[{row};{col}H"