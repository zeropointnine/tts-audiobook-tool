import logging
import os
import tempfile
from pathlib import Path
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
def stub_notification_sounds(monkeypatch):
    """Keep UI tests off real audio devices.

    A live callback stream can hang PortAudio's interpreter-exit cleanup, even
    with a daemon playback thread. Sound-selection tests install recording fakes.
    """
    from tts_audiobook_tool import app_support

    monkeypatch.setattr(app_support, "play_done_sound", lambda: None)
    monkeypatch.setattr(app_support, "play_fatal_gen_sound", lambda: None)


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


class _NoSleepTime:
    """``time`` module stand-in whose ``sleep`` returns immediately."""

    def __getattr__(self, name):
        import time
        return getattr(time, name)

    @staticmethod
    def sleep(_seconds):
        pass


@pytest.fixture(autouse=True)
def skip_ui_pauses(monkeypatch):
    """Drop the human-paced UI pauses: ``print_feedback``'s read-it pause
    (0.8 s, doubled for errors) and the hint "[...]" animation (2 s).

    The pause *values* are untouched, so tests that record what would have been
    slept (see test_print_feedback) still see the real durations.
    """
    from tts_audiobook_tool import util
    from tts_audiobook_tool.app_support import hints

    monkeypatch.setattr(util, "time", _NoSleepTime())
    monkeypatch.setattr(hints, "time", _NoSleepTime())


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


@pytest.fixture
def prefs_path() -> Path:
    """The prefs file location for this test (inside the isolated user dir)."""
    from tts_audiobook_tool.prefs import Prefs

    return Path(Prefs.get_file_path())


class FakeWorker:
    """Stands in for the worker process behind ``ModelWorker``.

    Generation and real-time playback submissions return ``"job"``, and
    ``drain_events`` hands out ``batches`` one list per call (``[]`` once
    empty, i.e. an idle worker). Assign or extend ``batches`` to feed events.
    """

    def __init__(self, monkeypatch) -> None:
        from tts_audiobook_tool.model_worker import ModelWorker

        self.batches: list[list] = []
        for method in ("submit_generation", "submit_realtime_playback"):
            monkeypatch.setattr(ModelWorker, method, staticmethod(lambda **_: "job"))
        monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(self._drain))

    def _drain(self, *_args, **_kwargs) -> list:
        return self.batches.pop(0) if self.batches else []


@pytest.fixture
def fake_worker(monkeypatch) -> FakeWorker:
    return FakeWorker(monkeypatch)


@pytest.fixture(autouse=True)
def initialize_tts_type_for_tests():
    # Real worker subprocesses initialize the same metadata catalog. Keep the
    # parent fingerprint/declarations genuine even when tests stub model loading.
    if not Tts._config_fingerprint:
        Tts.init_local_model_type()
    from tts_audiobook_tool.project_support.model_settings import REGISTRY
    original_metadata = {name: getattr(Tts, name) for name in (
        "_config_fingerprint", "_configured_definitions", "_audio_cpp_definitions", "_configured_runtime",
    )}
    original_registry = (dict(REGISTRY.bindings), dict(REGISTRY.legacy), dict(REGISTRY.parameter_bounds),
                         dict(REGISTRY.parameter_choices))
    original_model_catalog = (dict(TtsModelType._catalog), dict(TtsModelType._specs))
    had_type = hasattr(Tts, "_type")
    original_type = getattr(Tts, "_type", None)

    if not had_type or Tts._type is None:
        setattr(Tts, "_type", TtsModelType.require_by_id("none"))

    # The backend mode is a process invariant, probed from the remote-client
    # markers. The test venvs do not carry these markers, so pin
    # it to the probed (local) value here; tests that need remote-client mode
    # set Tts._backend_mode themselves and get it restored on teardown.
    had_mode = hasattr(Tts, "_backend_mode")
    original_mode = getattr(Tts, "_backend_mode", None)
    Tts._backend_mode = Tts._probe_backend_mode()
    binding_fields = {
        "_binding_issue": None,
        "_bound_project_type_id": "",
        "_remote_issue": "",
        "_selected_server_model_id": "",
        "_available_local_models": (),
        "_model_params": {},
    }
    original_binding_fields = {name: getattr(Tts, name) for name in binding_fields}
    for name, value in binding_fields.items():
        setattr(Tts, name, value)

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
        (REGISTRY.bindings, REGISTRY.legacy, REGISTRY.parameter_bounds,
         REGISTRY.parameter_choices) = original_registry
        TtsModelType._catalog, TtsModelType._specs = original_model_catalog
        for name, value in original_metadata.items():
            setattr(Tts, name, value)
        for name, value in original_binding_fields.items():
            setattr(Tts, name, value)
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