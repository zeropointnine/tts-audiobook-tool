import os
import platform
import subprocess
from pathlib import Path

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.util import get_package_dir, make_error_string, print_feedback


def get_chromium_info() -> tuple[str, str] | None:
    """Looks for a Chromium-based browser on the local machine."""
    system = platform.system()

    if system == "Windows":
        names_and_paths = [
            ("Chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            ("Chrome", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
            ("Chrome", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe")),
            ("Chromium", r"C:\Program Files\Chromium\Application\chrome.exe"),
            ("Chromium", r"C:\Program Files (x86)\Chromium\Application\chrome.exe"),
            ("Chromium", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Chromium", "Application", "chrome.exe")),
            ("Edge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        ]
    elif system == "Darwin":
        names_and_paths = [
            ("Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            ("Chromium", "/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    elif system == "Linux":
        names_and_paths = [
            ("Chrome", "/usr/bin/google-chrome"),
            ("Chrome", "/usr/bin/google-chrome-stable"),
            ("Chrome", "/opt/google/chrome/google-chrome"),
            ("Chromium", "/usr/bin/chromium"),
            ("Chromium", "/usr/bin/chromium-browser"),
            ("Chromium", "/usr/bin/chromium-browser-stable"),
            ("Chromium", "/snap/bin/chromium"),
        ]
    else:
        return None

    for item in names_and_paths:
        if os.path.exists(item[1]):
            return item

    return None


def get_chromium_user_data_dir() -> str:
    return os.path.join(app_paths.get_app_user_dir(), "chromium-user-data")

def launch_player_with_chromium(
    chromium_path: str,
    audio_file_path: str,
    user_data_dir: str,
) -> None:
    """Launch the local reader/player app in a Chromium-based browser."""
    if user_data_dir:
        os.makedirs(user_data_dir, exist_ok=True)
        if not os.path.exists(user_data_dir):
            print_feedback(f"Couldn't create browser user directory {user_data_dir}")
            return

    browser_flags = [
        "--allow-file-access-from-files",
        "--autoplay-policy=no-user-gesture-required",
    ]
    if user_data_dir:
        browser_flags.append(f"--user-data-dir={user_data_dir}")

    package_dir = get_package_dir()
    if package_dir:
        index_html_path = Path(package_dir).parent / "browser_player" / "index.html"
    else:
        index_html_path = Path("browser_player") / "index.html"
    if platform.system() == "Windows":
        index_html_url = index_html_path.absolute().as_uri()
    else:
        index_html_url = index_html_path.resolve().as_uri()
    browser_url = text_util.make_url_with_params(index_html_url, {"url": audio_file_path})

    command: list[str] = [chromium_path]
    command.extend(browser_flags)
    command.append(browser_url)

    try:
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print_feedback(f"Launched process:\n{command}")
    except (FileNotFoundError, Exception) as e:
        print_feedback(make_error_string(e), is_error=True)
