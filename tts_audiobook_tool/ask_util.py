import sys
from tts_audiobook_tool.util import *


class AskUtil:
    """
    "Ask" == "get text input from user"
    """

    # Gets set to False if using readchar in ask_hotkey() fails
    is_readchar = True

    @staticmethod
    def ask(message: str="", lower: bool=True, extra_line: bool=True) -> str:
        """
        App-standard way of getting user line input.
        Prints extra line after the input by default.
        """

        if not DEV:
            clear_input_buffer()

        message = f"{message}{COL_INPUT}"
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
    def ask_enter_to_continue(prefix_line: str="") -> None:
        message = "Press enter: "
        if prefix_line:
            message = f"{prefix_line}\n{message}"
        if AskUtil.is_readchar:
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
        if AskUtil.is_readchar:
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
        Uses `readchar` to block and return next keypress.
        Falls back to vanilla input()-based input if necessary.
        """
        if not AskUtil.is_readchar:
            return AskUtil.ask_hotkey_vanilla(message, lower)

        try:
            import readchar
        except ImportError:
            AskUtil.is_readchar = False
            return AskUtil.ask_hotkey_vanilla(message, lower)

        if not sys.stdin.isatty():
            AskUtil.is_readchar = False
            return AskUtil.ask_hotkey_vanilla(message, lower)

        if message:
            printt(message.strip() + " ", end="") # cursor is positioned to right of message

        if not DEV:
            clear_input_buffer()

        try:
            s = readchar.readkey()
        except KeyboardInterrupt:
            # This happens on control-c in windows but not linux/macos
            return "\x03" # ie, control-c
        except Exception as e:
            AskUtil.is_readchar = False
            return AskUtil.ask_hotkey_vanilla("", lower)

        if False:
            print("Hotkey: ", repr(s)) # print escaped string

        if message:
            printt()

        if lower:
            s = s.lower()

        return s

    @staticmethod
    def ask_hotkey_vanilla(message: str="", lower: bool=True) -> str:
        inp = AskUtil.ask(message, lower, extra_line=True)
        if inp:
            inp = inp[0]
        return inp

    @staticmethod
    def ask_file_path(
            console_message: str,
            requestor_title: str,
            filetypes: list[tuple[str, str]] = [],
            initialdir: str=""
    ) -> str:
        try:
            from tkinter import filedialog
            printt(console_message)
            result = filedialog.askopenfilename(title=requestor_title, filetypes=filetypes, initialdir=initialdir)
            printt(result)
            printt()
            return result
        except Exception as e:
            return AskUtil.ask_path_input(console_message)

    @staticmethod
    def ask_dir_path(
            console_message: str,
            ui_title: str,
            initialdir: str = "",
            mustexist: bool = True,
    ) -> str:
        try:
            from tkinter import filedialog
            printt(console_message)
            result = filedialog.askdirectory(title=ui_title, initialdir=initialdir, mustexist=mustexist) # fyi, mustexist doesn't rly do anything on Windows
            if result:
                printt(result)
            printt()
            return result
        except Exception as e:
            return AskUtil.ask_path_input(console_message)

    @staticmethod
    def ask_path_input(message: str="") -> str:
        """
        Get file/directory path, strip outer quotes
        """
        printt(message)
        inp = AskUtil.ask("")
        return strip_quotes_from_ends(inp)
    
    @staticmethod
    def is_shell_gui_gtk_based() -> bool:
        desktop_env = get_desktop_environment()
        is_gtk = is_gtk_based(desktop_env)
        return is_gtk

# ---

def get_desktop_environment():
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

def is_gtk_based(desktop_string):
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
