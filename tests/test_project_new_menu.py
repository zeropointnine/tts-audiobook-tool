"""
Regression coverage for the "Enter the path to an empty directory:" prompt.

A path to an existing *file* must be rejected with a clean error string, not
raise NotADirectoryError out of State.make_and_set_new_project. The plain
"Make new project" flow has no try/except around that call, so before the
guard existed it crashed the app; the ABR flow swallowed it into an errno
message.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.constants import PROJECT_SOUND_SEGMENTS_SUBDIR
from tts_audiobook_tool.menus.project_new_menu import ProjectNewMenu
from tts_audiobook_tool.menus import project_new_menu as project_new_menu_module
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@contextmanager
def _quiet_project_setter():
    """Silences the model/whitelist side effects of State.project = ..."""
    with patch("tts_audiobook_tool.tts.Tts.set_model_params_using_project"), \
            patch("tts_audiobook_tool.tts.Tts.get_type", return_value=TtsModelType.NONE), \
            patch("tts_audiobook_tool.state.Whitelist"):
        yield


def _make_state() -> State:
    state = State.for_worker(Prefs(project_dir="", stt_variant=SttVariant.DISABLED))
    with _quiet_project_setter():
        state.project = Project(dir_path="")
    return state


def _stub_prompt(monkeypatch, path: str) -> None:
    monkeypatch.setattr(ask, "ask_dir_path", lambda **_kwargs: path)


def _collect_exits(monkeypatch) -> tuple[list[str], list[int]]:
    """Records the two ways the ABR flow can end: an error, or a press-enter wait."""
    errors: list[str] = []
    continues: list[int] = []
    monkeypatch.setattr(ask, "ask_error", lambda message: errors.append(message))
    monkeypatch.setattr(ask, "ask_enter_to_continue", lambda *_args, **_kwargs: continues.append(1))
    return errors, continues


def _collect_feedback(monkeypatch) -> list[str]:
    feedback: list[str] = []
    monkeypatch.setattr(project_new_menu_module, "print_feedback", lambda message, **_kwargs: feedback.append(message))
    return feedback


# --- State.make_and_set_new_project ---

def test_make_and_set_new_project_rejects_a_file_path(tmp_path):
    state = _make_state()
    file_path = tmp_path / "not_a_directory.flac"
    file_path.write_text("")

    error = state.make_and_set_new_project(str(file_path))

    assert error == f"Not a directory: {file_path}"
    assert list(tmp_path.iterdir()) == [file_path]


def test_make_and_set_new_project_rejects_a_non_empty_directory(tmp_path):
    state = _make_state()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("")

    assert state.make_and_set_new_project(str(occupied)) == "Directory is not empty"


def test_make_and_set_new_project_creates_a_missing_directory(tmp_path):
    state = _make_state()
    target = tmp_path / "new_project"

    with _quiet_project_setter():
        error = state.make_and_set_new_project(str(target))

    assert error == ""
    assert (target / PROJECT_SOUND_SEGMENTS_SUBDIR).is_dir()


# --- Menu flows that prompt for the directory ---

def test_make_new_project_reports_file_path_instead_of_raising(monkeypatch, tmp_path):
    state = _make_state()
    file_path = tmp_path / "not_a_directory.flac"
    file_path.write_text("")
    errors, _ = _collect_exits(monkeypatch)
    _stub_prompt(monkeypatch, str(file_path))

    assert ProjectNewMenu.make_new_project(state) is False
    assert errors == [f"Not a directory: {file_path}"]
    assert list(tmp_path.iterdir()) == [file_path]


def test_make_new_project_using_abr_reports_file_path_instead_of_errno(monkeypatch, tmp_path):
    state = _make_state()
    file_path = tmp_path / "not_a_directory.flac"
    file_path.write_text("")
    errors, _ = _collect_exits(monkeypatch)
    _stub_prompt(monkeypatch, str(file_path))

    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: str(file_path))
    monkeypatch.setattr(
        project_new_menu_module.AppMetadata,
        "load_from_file",
        staticmethod(lambda _path: SimpleNamespace(project_snapshot={"settings": {}})),
    )

    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert errors == [f"Not a directory: {file_path}"]
    assert list(tmp_path.iterdir()) == [file_path]


# --- Cancellation of the path prompts ---

def test_abr_flow_reports_cancelled_when_the_abr_prompt_is_cancelled(monkeypatch, tmp_path):
    state = _make_state()
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    _, continues = _collect_exits(monkeypatch)
    feedback = _collect_feedback(monkeypatch)
    _stub_prompt(monkeypatch, str(dest_dir))
    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: "")

    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert feedback == ["\nCancelled"]
    assert continues == []
    assert list(dest_dir.iterdir()) == []


def test_abr_flow_reports_cancelled_when_the_directory_prompt_is_cancelled(monkeypatch):
    state = _make_state()
    _, continues = _collect_exits(monkeypatch)
    feedback = _collect_feedback(monkeypatch)
    _stub_prompt(monkeypatch, "")

    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert feedback == ["\nCancelled"]
    assert continues == []


def test_abr_flow_still_waits_for_enter_on_a_validation_error(monkeypatch, tmp_path):
    state = _make_state()
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    errors, continues = _collect_exits(monkeypatch)
    feedback = _collect_feedback(monkeypatch)
    _stub_prompt(monkeypatch, str(dest_dir))
    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: str(tmp_path / "notes.txt"))

    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert errors == ["Please select a .flac, .m4a, or .m4b file"]
    assert continues == [1]
    assert feedback == []


# --- Model-mismatch FYI for a project cloned from an ABR file ---

@contextmanager
def _runtime_model(model_type):
    """Holds the runtime TTS model steady for the duration of a menu flow."""
    with patch("tts_audiobook_tool.tts.Tts.set_model_params_using_project"), \
            patch("tts_audiobook_tool.tts.Tts.get_type", return_value=model_type), \
            patch("tts_audiobook_tool.state.Whitelist"):
        yield


def _collect_shown_hints(monkeypatch) -> list:
    """Captures the hints the flow shows, bypassing the prefs-gated ones."""
    shown: list = []
    monkeypatch.setattr(
        project_new_menu_module,
        "hints",
        SimpleNamespace(
            show_hint=lambda hint, **_kwargs: shown.append(hint) or True,
            show_hint_if_necessary=lambda *_args, **_kwargs: True,
        ),
    )
    return shown


def _run_abr_import(monkeypatch, tmp_path, state, source_model_id: str) -> tuple[list, list[str]]:
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    _stub_prompt(monkeypatch, str(dest_dir))
    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: str(tmp_path / "book.abr.flac"))
    monkeypatch.setattr(
        project_new_menu_module.AppMetadata,
        "load_from_file",
        staticmethod(lambda _path: SimpleNamespace(project_snapshot={
            "dir_path": str(tmp_path / "source_project"),
            "current_model_type": source_model_id,
        })),
    )
    errors, _ = _collect_exits(monkeypatch)
    shown = _collect_shown_hints(monkeypatch)

    assert ProjectNewMenu.make_new_project_using_abr(state) is True, errors
    return shown, errors


def test_abr_import_shows_model_mismatch_fyi_for_the_source_model(monkeypatch, tmp_path):
    state = _make_state()

    with _runtime_model(TtsModelType.MIRA):
        shown, _errors = _run_abr_import(
            monkeypatch, tmp_path, state, TtsModelType.CHATTERBOX.value.id
        )

    assert [hint.heading for hint in shown] == ["FYI"]
    assert "Chatterbox TTS" in shown[0].text
    assert "differs from the model currently in use" in shown[0].text


def test_abr_import_shows_no_model_mismatch_fyi_for_the_current_model(monkeypatch, tmp_path):
    state = _make_state()

    with _runtime_model(TtsModelType.MIRA):
        shown, _errors = _run_abr_import(
            monkeypatch, tmp_path, state, TtsModelType.MIRA.value.id
        )

    assert shown == []


def test_abr_import_consumes_the_pending_mismatch_so_the_main_menu_stays_quiet(monkeypatch, tmp_path):
    state = _make_state()

    with _runtime_model(TtsModelType.MIRA):
        _run_abr_import(monkeypatch, tmp_path, state, TtsModelType.CHATTERBOX.value.id)

    assert state.pending_model_mismatch_name == ""
    assert state.take_model_mismatch_name() == ""


# --- Console-fallback path normalization ---

def _no_gui_dialog(*_args, **_kwargs):
    raise RuntimeError("no display")


def test_ask_dir_path_expands_leading_tilde(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("tkinter.filedialog.askdirectory", _no_gui_dialog)
    monkeypatch.setattr(ask, "ask_path_input", lambda *_args, **_kwargs: "~/new_project")

    assert ask.ask_dir_path("Enter the path to an empty directory:", "Select empty directory") == (
        str(home / "new_project")
    )


def test_ask_file_path_expands_leading_tilde(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", _no_gui_dialog)
    monkeypatch.setattr(ask, "ask_path_input", lambda *_args, **_kwargs: "~/book.flac")

    assert ask.ask_file_path("Enter the path to an audio file:", "Select audio file") == (
        str(home / "book.flac")
    )
