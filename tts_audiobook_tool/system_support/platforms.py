import os
import platform
import subprocess
from pathlib import Path

from tts_audiobook_tool.util import make_error_string


def has_gui() -> bool:
    """Check whether the current host appears to have a desktop GUI available."""
    system = platform.system()
    if system == "Linux":
        return "DISPLAY" in os.environ
    if system == "Darwin":
        return True
    if system == "Windows":
        return True
    return False


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False

    checks = [
        "microsoft" in platform.uname().release.lower(),
        "WSL_DISTRO_NAME" in os.environ,
        os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"),
    ]
    return any(checks)


def open_directory(dir_path: str) -> str:
    """
    Robust method to open a directory in the system file explorer.
    Returns empty string on success, otherwise an error string.
    """
    try:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            return "Directory does not exist"
        if not path_obj.is_dir():
            return "Path is not a directory"

        if not has_gui():
            return "No recognized GUI environment detected"
        if is_wsl():
            return "Unsupported for WSL"

        abs_path = str(path_obj.resolve())
        system = platform.system()

        if system == "Windows":
            try:
                os.startfile(abs_path)  # type: ignore[attr-defined]
            except Exception:
                subprocess.Popen(["explorer", abs_path])
        elif system == "Darwin":
            subprocess.Popen(["open", abs_path])
        else:
            try:
                subprocess.Popen(["xdg-open", abs_path])
            except FileNotFoundError:
                managers = ["nautilus", "dolphin", "thunar", "pcmanfm", "konqueror"]
                for manager in managers:
                    try:
                        subprocess.Popen([manager, abs_path])
                        break
                    except FileNotFoundError:
                        continue
                else:
                    return "No file manager found"

        return ""
    except Exception as e:
        return make_error_string(e)
