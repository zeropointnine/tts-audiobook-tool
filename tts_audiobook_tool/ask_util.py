import os
import sys
from typing import Callable
from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_types import Saveable
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.util import *


def clear_input_buffer() -> None:
    """Use before input reads to prevent buffered keystrokes from being consumed."""
    try:
        import msvcrt
        while msvcrt.kbhit(): # type: ignore
            msvcrt.getch() # type: ignore
    except ImportError:
        if not sys.stdin.isatty():
            return
        import termios
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH) # type: ignore
        except termios.error:
            return

class AskUtil:
    """
    "Ask" == "get text input from user"
    """

    # Whether single-key hotkey input is currently available.
    # This starts from the TTY capability check, and can later be downgraded
    # to False if a low-level hotkey read fails unexpectedly.
    can_hotkey = sys.stdin.isatty()

    @staticmethod
    def ask(message: str="", lower: bool=True, extra_line: bool=True) -> str:
        """
        App-standard way of getting user line input.
        Prints extra line after the input by default.
        """

        if not DEV:
            clear_input_buffer()

        message = f"{message}{Ansi.RESET}{COL_INPUT}"
        try:
            inp = input(message).strip()
        except (ValueError, EOFError) as e:
            return ""
        if lower:
            inp = inp.lower()
        print(Ansi.RESET, end="")
        if extra_line:
            printt()
        return inp

    @staticmethod
    def ask_enter_to_continue(value: str="", is_replacement: bool=False) -> None:

        if is_replacement:
            message = value
        else:
            message = f"{value}\nPress enter: "

        if AskUtil.can_hotkey:
            while True:
                key = AskUtil.ask_hotkey(message)
                if key in ["\r", "\n"]:
                    printt()
                    return
                message = ""
        else:
            AskUtil.ask(message)

    @staticmethod
    def ask_confirm(message: str="") -> bool:
        if not message:
            message = f"Press {make_hotkey_string('Y')} to confirm: "
        inp = AskUtil.ask_hotkey(message)
        if AskUtil.can_hotkey:
            printt()
        return inp == "y"

    @staticmethod
    def ask_error(error_message: str) -> None:
        printt(f"{COL_ERROR}{error_message}")
        printt()
        AskUtil.ask_enter_to_continue()

    @staticmethod
    def ask_hotkey(message: str="", lower: bool=True) -> str:
        """
        Blocks until a single hotkey press is read.
        Falls back to vanilla input()-based input if necessary.
        """
        if not AskUtil.can_hotkey:
            return AskUtil.ask_hotkey_vanilla(message, lower)

        if message:
            printt(message.strip() + " ", end="") # cursor is positioned to right of message

        if not DEV:
            clear_input_buffer()

        try:
            if os.name == 'nt':
                s = AskUtil.read_hotkey_windows()
            else:
                s = AskUtil.read_hotkey_posix()
        except KeyboardInterrupt:
            # This happens on control-c in windows but not linux/macos
            return "\x03" # ie, control-c
        except Exception as e:
            AskUtil.can_hotkey = False
            return AskUtil.ask_hotkey_vanilla("", lower)

        if False:
            print("Hotkey: ", repr(s)) # print escaped string

        if message:
            printt()

        if lower:
            s = s.lower()

        return s

    @staticmethod
    def ask_multiline() -> str:
        """ App standard way of getting multi-line string input from user """
        return sys.stdin.read().strip()

    @staticmethod
    def read_hotkey_windows(block: bool=True) -> str | None:
        """
        Read one hotkey on Windows and normalize common special keys.

        When `block` is False, returns None if no key is waiting.
        """
        import msvcrt

        if not block and not msvcrt.kbhit(): # type: ignore
            return None

        ch = msvcrt.getwch() # type: ignore
        if ch in ("\x00", "\xe0"):
            special = msvcrt.getwch() # type: ignore
            mapped = {
                "H": "\x1b[A",
                "P": "\x1b[B",
                "K": "\x1b[D",
                "M": "\x1b[C",
                "S": "\x1b[3~",
                "G": "\x1b[H",
                "O": "\x1b[F",
                "I": "\x1b[5~",
                "Q": "\x1b[6~",
            }.get(special)
            if mapped is not None:
                return mapped
            return ch + special

        if ch == "\r":
            return "\r"
        if ch == "\x08":
            return "\x08"
        if ch == "\x03":
            return "\x03"
        if ch == "\x1b":
            return "\x1b"
        return ch

    @staticmethod
    def read_hotkey_posix() -> str:
        """
        Read one hotkey on POSIX terminals.

        This treats a lone Escape keypress as Escape after a short timeout
        instead of always blocking for the next byte in a possible escape
        sequence.
        """
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)

            first = os.read(fd, 1)
            if first != b"\x1b":
                return first.decode("utf-8", errors="replace")

            sequence = first
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                return "\x1b"

            for _ in range(8):
                if not select.select([sys.stdin], [], [], 0)[0]:
                    break
                sequence += os.read(fd, 1)

            return sequence.decode("utf-8", errors="replace")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    @staticmethod
    def ask_hotkey_vanilla(message: str="", lower: bool=True) -> str:
        inp = AskUtil.ask(message, lower, extra_line=True)
        if inp:
            inp = inp[0]
        return inp

    @staticmethod
    def ask_file_path(
            console_message: str,
            dialog_title: str,
            filetypes: list[tuple[str, str]] = [],
            initialdir: str=""
    ) -> str:
        """
        Gets a file path string from user using either gui file requestor or input().
        """
        try:
            from tkinter import filedialog
            printt(console_message)
            path = filedialog.askopenfilename(title=dialog_title, filetypes=filetypes, initialdir=initialdir)
            did_tk = True
            if isinstance(path, tuple):
                # Can return an empty tuple on cancel (Linux Mint Cinnamon)
                path = ""
        except Exception as e:
            path = AskUtil.ask_path_input(console_message)
            did_tk = False

        if not path:
            return ""
        path = os.path.normpath(os.path.abspath(path))
        
        if did_tk:
            printt(path)
        printt()
        return path

    @staticmethod
    def ask_dir_path(
            console_message: str,
            dialog_title: str,
            initialdir: str = "",
            mustexist: bool = True,
    ) -> str:
        """
        Gets a dir path string from user using either gui file requestor or input().
        """
        try:
            # FYI: GTK-based folder requestor dialog has no obvious "new folder" functionality,
            # but will return a non-existing directory path if manually entered; not ideal but
            from tkinter import filedialog
            printt(console_message)
            path = filedialog.askdirectory(title=dialog_title, initialdir=initialdir, mustexist=mustexist) # fyi, mustexist doesn't rly do anything on Windows
            did_tk = True
            if isinstance(path, tuple):
                # Can return an empty tuple on cancel (Linux Mint Cinnamon)
                path = ""
        except Exception as e:
            path = AskUtil.ask_path_input(console_message)
            did_tk = False

        if not path:
            return ""
        path = os.path.normpath(os.path.abspath(path))
        
        if did_tk:
            printt(path)
        printt()
        return path

    @staticmethod
    def ask_path_input(message: str="") -> str:
        """
        Get file/directory path, strip outer quotes
        """
        printt(message)
        inp = AskUtil.ask("")
        return text_util.strip_quotes_around_path_string(inp)

    @staticmethod
    def is_shell_gui_gtk_based() -> bool:
        desktop_env = get_desktop_environment()
        is_gtk = is_gtk_based(desktop_env)
        return is_gtk

    @staticmethod
    def ask_number(
        saveable: Saveable,
        attr: str,
        prompt: str,
        min_value: float,
        max_value: float,
        default_value: float,
        success_prefix: str,
        is_int: bool=False
    ) -> None:
        """
        """

        # Type checking for Prefs or Project, workaround for circular import
        # Function would work on any "object" regardless but yea
        from tts_audiobook_tool.prefs import Prefs
        from tts_audiobook_tool.project import Project
        if not isinstance(saveable, Project) and not isinstance(saveable, Prefs):
            raise ValueError(f"Not Project or Prefs: {saveable}")

        if not hasattr(saveable, attr):
            raise ValueError(f"No such attribute {attr}")

        if is_int:
            min_value = int(min_value)
            max_value = int(max_value)
            default_value = int(default_value)

        prompt = prompt.strip()
        if not prompt.endswith(":"):
            prompt += ":"
        prompt += " " + f"{COL_DIM}(valid range: {min_value}-{max_value}; default: {default_value})"
        printt(prompt)
        value = AskUtil.ask()
        if not value:
            return
        try:
            # fyi, always cast to float bc "int(5.1)"" throws exception in 3.11 seems like
            value = float(value)
        except Exception as e:
            print_feedback("Bad value", is_error=True)
            return
        if is_int:
            value = int(value)
        if not (min_value <= value <= max_value):
            print_feedback("Out of range", is_error=True)
            return

        setattr(saveable, attr, value)
        if not isinstance(saveable, Project):
            saveable.save()

        print_feedback(success_prefix, str(value))

    @staticmethod
    def ask_number_and_save(
        saveable: Saveable,
        prompt: str,
        lb: float,
        ub: float,
        project_attr_name: str,
        success_prefix: str,
        is_int: bool=False
    ) -> None:

        from tts_audiobook_tool.project import Project

        if not hasattr(saveable, project_attr_name):
            raise ValueError(f"No such attribute {project_attr_name}")

        if is_int:
            lb = int(lb)
            ub = int(ub)

        value = AskUtil.ask(prompt.strip() + " ")
        if not value:
            return
        try:
            # fyi, always cast to float bc "int(5.1)"" throws exception in 3.11 seems like
            value = float(value)
        except Exception as e:
            print_feedback("Bad value", is_error=True)
            return
        if is_int:
            value = int(value)
        if not (lb <= value <= ub):
            print_feedback("Out of range", is_error=True)
            return

        setattr(saveable, project_attr_name, value)
        if not isinstance(saveable, Project):
            saveable.save()
        print_feedback(success_prefix, str(value))

    @staticmethod
    def ask_string_and_save(
        saveable: Saveable,
        prompt_line: str,
        project_attr_name: str,
        success_prefix: str,
        loop_on_error: bool=False,
        validator: Callable[[str], str] | None = None
    ) -> None:
        """
        Helper to ask for a string value and save it to the project.
        :param validator: Takes in the user input string and returns error string if invalid (optional)
        """
        from tts_audiobook_tool.project import Project

        if not hasattr(saveable, project_attr_name):
            raise ValueError(f"No such attribute {project_attr_name}")

        while True:
            printt(prompt_line)
            value = AskUtil.ask(lower=False)
            if not value:
                return            
            if validator:
                err = validator(value)
                if err:
                    print_feedback(err, is_error=True)
                    if loop_on_error:
                        continue
                    else:
                        return
                break                    

        setattr(saveable, project_attr_name, value)
        if not isinstance(saveable, Project):
            saveable.save()
        print_feedback(success_prefix, value)

# ---

def get_desktop_environment() -> str | None:
    """
    Checks standard environment variables to determine the desktop environment.
    Returns a string like 'GNOME', 'Cinnamon', 'KDE', etc., or None if not found.
    """
    # Modern standard: XDG_CURRENT_DESKTOP
    # This is the most reliable variable.
    desktop = os.environ.get('XDG_CURRENT_DESKTOP')
    if desktop:
        return desktop

    # Fallback for older systems or different display managers
    desktop = os.environ.get('DESKTOP_SESSION')
    if desktop:
        return desktop

    # Another common fallback
    desktop = os.environ.get('GDMSESSION')
    if desktop:
        return desktop

    return None

def is_gtk_based(desktop_string: str | None) -> bool:
    """
    Checks if a desktop environment string corresponds to a GTK-based DE.
    Cinnamon, MATE, and XFCE are all GTK-based.
    """
    if not desktop_string:
        return False

    # List of common GTK desktop identifiers (case-insensitive)
    gtk_desktops = ['gnome', 'cinnamon', 'mate', 'xfce', 'budgie', 'pantheon']

    for de in gtk_desktops:
        if de in desktop_string.lower():
            return True

    return False
