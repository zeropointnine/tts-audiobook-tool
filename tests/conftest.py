import logging
import os
import tempfile
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.l import L
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.fixture(autouse=True)
def stop_project_directory_watchers(monkeypatch):
    """Release each test's project observers instead of exhausting inotify."""
    from tts_audiobook_tool.project_support.project_sound_segments import ProjectSoundSegments

    original_init = ProjectSoundSegments.__init__
    observers = []

    def tracked_init(self, project):
        original_init(self, project)
        if self.observer.is_alive():
            observers.append(self.observer)

    monkeypatch.setattr(ProjectSoundSegments, "__init__", tracked_init)
    yield
    for observer in observers:
        observer.stop()
    for observer in observers:
        observer.join(timeout=2)


@pytest.fixture(autouse=True)
def initialize_app_logger():
    """Ensure ``L.logger`` exists.

    Production calls ``L.init()`` during app startup; tests that exercise code
    paths which log (directly or via ``call_after_refresh``) would otherwise
    fail on the unset attribute.
    """
    if getattr(L, "logger", None) is None:
        L.logger = logging.getLogger("tests")
    yield


@pytest.fixture(autouse=True)
def stub_system_sleep_inhibitor(monkeypatch):
    """Keeps the suite off the host's power-management stack.

    ``prevent_system_sleep`` otherwise opens a real session inhibit (D-Bus /
    systemd / SetThreadExecutionState) every time a generation, concat, or
    real-time playback entry point runs. Tests that assert the wiring replace
    ``system_sleep._wakepy_keep`` with their own recording fake.
    """

    class FakeMode:
        active = True
        active_method = SimpleNamespace(name="tests")

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    class FakeKeep:
        @staticmethod
        def running(**_kwargs):
            return FakeMode()

    from tts_audiobook_tool.app_support import system_sleep

    monkeypatch.setattr(system_sleep, "_wakepy_keep", FakeKeep())
    yield


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

    # Production calls init_local_model_type() at startup, which finalizes the
    # catalog and marks it initialized. Tests that drive menus or the worker
    # would otherwise have ModelWorker.start() re-probe the venv and overwrite
    # the model type the test just chose.
    had_catalog = hasattr(Tts, "_catalog_initialized")
    original_catalog = getattr(Tts, "_catalog_initialized", False)
    Tts._catalog_initialized = True

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

        if had_catalog:
            setattr(Tts, "_catalog_initialized", original_catalog)
        else:
            setattr(Tts, "_catalog_initialized", original_catalog)