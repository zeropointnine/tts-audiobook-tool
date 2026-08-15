import os
import tempfile

import pytest

from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.fixture(autouse=True)
def isolate_app_user_dir(monkeypatch):
    """
    Redirects the app's user directory to a per-test temp directory so that no test can
    read or write real user files (e.g. the prefs file) in the actual home directory.
    Uses its own TemporaryDirectory (rather than tmp_path) so it doesn't pollute the
    directory that tests assert against directly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        user_dir = os.path.join(temp_dir, "app-user-dir")
        os.makedirs(user_dir, exist_ok=True)
        monkeypatch.setattr(app_paths, "get_app_user_dir", lambda: user_dir)
        yield


@pytest.fixture(autouse=True)
def initialize_tts_type_for_tests():
    had_type = hasattr(Tts, "_type")
    original_type = getattr(Tts, "_type", None)

    if not had_type or Tts._type is None:
        setattr(Tts, "_type", TtsModelType.NONE)

    try:
        yield
    finally:
        if had_type:
            setattr(Tts, "_type", original_type)
        elif hasattr(Tts, "_type"):
            delattr(Tts, "_type")
