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

    # The backend mode is a process invariant, probed from the SGL-Omni
    # sentinel package. The test venvs do not carry the sentinel, so pin
    # it to the probed (local) value here; tests that need SGL-Omni mode
    # set Tts._backend_mode themselves and get it restored on teardown.
    had_mode = hasattr(Tts, "_backend_mode")
    original_mode = getattr(Tts, "_backend_mode", None)
    Tts._backend_mode = Tts._probe_backend_mode()

    try:
        yield
    finally:
        if had_type:
            setattr(Tts, "_type", original_type)
        elif hasattr(Tts, "_type"):
            delattr(Tts, "_type")

        if had_mode:
            setattr(Tts, "_backend_mode", original_mode)
        elif hasattr(Tts, "_backend_mode"):
            delattr(Tts, "_backend_mode")
