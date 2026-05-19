import os
import tempfile

from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.util import *


def get_app_temp_dir() -> str:
    dir = os.path.join(tempfile.gettempdir(), APP_TEMP_SUBDIR)
    Path(dir).mkdir(exist_ok=True)
    return dir


def get_app_user_dir() -> str:
    dir = os.path.join(Path.home(), APP_USER_SUBDIR)
    Path(dir).mkdir(exist_ok=True)
    return dir


def get_chromium_user_data_dir() -> str:
    return os.path.join(get_app_user_dir(), CHROME_USER_DATA_DIR_NAME)


def get_temp_file_path_by_hash(hash: str) -> str:
    """
    Returns file path of item in the app temp directory or empty string
    """
    dir = get_app_temp_dir()
    if not dir:
        return ""
    items = os.listdir(dir)
    for item in items:
        if hash in item:
            return os.path.join(dir, item)
    return ""


def path_insert_tag(file_path: str, tag: str) -> str:
    """
    Eg, "[one] [two] hello.flac" -> "[one] [two] [newtag] hello.flac"
    """
    path = Path(file_path)
    stem = path.stem
    i = stem.rfind("]") + 1
    substring = f"[{tag}]"
    if i > 0:
        substring = " " + substring
    else:
        substring = substring + " "
    new_stem = stem[:i] + substring + stem[i:]
    new_path = path.with_stem(new_stem)
    return str(new_path)