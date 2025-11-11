import os
import subprocess
import platform
from pathlib import Path

from tts_audiobook_tool.util import *

class DirOpenUtil:

    @staticmethod
    def open(dir_path: str) -> str:
        """
        Robust method to open directory in system file explorer
        Returns empty string or error string
        """
        try:
            # Validate and normalize path
            path_obj = Path(dir_path)
            if not path_obj.exists():
                return "Directory does not exist"
            if not path_obj.is_dir():
                return "Path is not a directory"

            abs_path = str(path_obj.resolve())

            # Platform-specific handling
            system = platform.system()

            if system == "Windows":
                # Two methods for Windows
                try:
                    os.startfile(abs_path) # type: ignore
                except:
                    subprocess.Popen(['explorer', abs_path])

            elif system == "Darwin":
                subprocess.Popen(['open', abs_path])

            else:  # Linux and other Unix-like systems
                # Try xdg-open first, then common file managers
                try:
                    subprocess.Popen(['xdg-open', abs_path])
                except FileNotFoundError:
                    managers = ['nautilus', 'dolphin', 'thunar', 'pcmanfm', 'konqueror']
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
