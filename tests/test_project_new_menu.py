"""
Regression coverage for the "Enter the path to an empty directory:" prompt.

A path to an existing *file* must be rejected with a clean error string, not
raise NotADirectoryError out of State.make_and_set_new_project. The plain
"Make new project" flow has no try/except around that call, so before the
guard existed it crashed the app; the ABR flow swallowed it into an errno
message.
"""

from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.constants import PROJECT_SOUND_SEGMENTS_SUBDIR, PROJECT_VOICE_SUBDIR
from tts_audiobook_tool.menus.project_new_menu import ProjectNewMenu
from tts_audiobook_tool.menus import project_new_menu as project_new_menu_module
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State


@contextmanager
def _quiet_project_setter():
    """Silences the model/whitelist side effects of State.project = ..."""
    with patch("tts_audiobook_tool.tts.Tts.set_model_params_using_project"), \
            patch("tts_audiobook_tool.tts.Tts.bind_project"), \
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
        staticmethod(lambda _path: SimpleNamespace(project_snapshot={"version": 2, "max_words": 50})),
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


# --- Project cloned from an ABR file ---

def _run_abr_import(
    monkeypatch,
    tmp_path,
    state,
    source_dir: str | None = None,
) -> list[str]:
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    _stub_prompt(monkeypatch, str(dest_dir))
    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: str(tmp_path / "book.abr.flac"))
    monkeypatch.setattr(
        project_new_menu_module.AppMetadata,
        "load_from_file",
        staticmethod(lambda _path: SimpleNamespace(project_snapshot={
            "dir_path": source_dir if source_dir is not None else str(tmp_path / "source_project"),
            "language_code": "en",
        })),
    )
    errors, _ = _collect_exits(monkeypatch)

    assert ProjectNewMenu.make_new_project_using_abr(state) is True, errors
    return errors


# --- Supporting files for a project cloned from another computer ---

def test_abr_import_explains_a_source_project_that_is_not_here(monkeypatch, tmp_path, capsys):
    """
    The settings travelled without the voice samples they refer to. Saying so is
    the difference between a confusingly empty project and a user who knows what
    still has to be copied over.
    """
    state = _make_state()

    with _quiet_project_setter():
        _run_abr_import(monkeypatch, tmp_path, state)

    assert "not a project directory on this computer" in capsys.readouterr().out


def test_abr_import_stays_quiet_when_the_source_project_is_here(monkeypatch, tmp_path, capsys):
    state = _make_state()
    source_dir = tmp_path / "source_project"
    (source_dir / PROJECT_VOICE_SUBDIR).mkdir(parents=True)

    with _quiet_project_setter():
        _run_abr_import(monkeypatch, tmp_path, state, str(source_dir))

    assert "not a project directory on this computer" not in capsys.readouterr().out


# --- Old and damaged ABR settings snapshots ---


def _stub_abr_snapshot(monkeypatch, tmp_path, snapshot):
    _stub_prompt(monkeypatch, str(tmp_path / "dest"))
    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: str(tmp_path / "book.abr.flac"))
    metadata = AppMetadata.get_from_json_string(json.dumps({
        "version": 2,
        "text_segments": [{"text": "Hello", "time_start": 0, "time_end": 1}],
        "project_snapshot": snapshot,
    }))
    assert isinstance(metadata, AppMetadata)
    monkeypatch.setattr(project_new_menu_module.AppMetadata, "load_from_file", lambda _path: metadata)


def test_abr_import_migrates_old_flat_settings_before_selecting(monkeypatch, tmp_path):
    state = _make_state()
    snapshot = {
        "version": 2,
        "fish_s2_voice_file_name": ["narrator.flac"],
        "fish_s2_temperature": 0.75,
        "fish_s2_seed": 42,
        "fish_s2_server_concurrent_requests": 3,
    }
    _stub_abr_snapshot(monkeypatch, tmp_path, snapshot)
    errors, _ = _collect_exits(monkeypatch)
    with _quiet_project_setter():
        assert ProjectNewMenu.make_new_project_using_abr(state) is True
    assert not errors
    assert state.project.get_model_setting("fish_s2_sglomni", "file_name") == ["narrator.flac"]
    assert state.project.get_model_setting("fish_s2_sglomni", "temperature") == 0.75
    assert state.project.get_model_setting("fish_s2_local", "seed") == 42
    assert state.project.get_model_setting("fish_s2_sglomni", "concurrent_requests") == 3
    assert state.prefs.project_dir == str(tmp_path / "dest")


def test_abr_import_rejects_junk_snapshot_without_creating_project(monkeypatch, tmp_path):
    state = _make_state()
    previous = state.project
    _stub_abr_snapshot(monkeypatch, tmp_path, {"settings": {}})
    errors, _ = _collect_exits(monkeypatch)
    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert "no recognizable project settings" in errors[0]
    assert not (tmp_path / "dest").exists()
    assert state.project is previous and state.prefs.project_dir == ""


def test_abr_import_without_snapshot_refuses_before_creating_project(monkeypatch, tmp_path, capsys):
    state = _make_state()
    _stub_abr_snapshot(monkeypatch, tmp_path, {})
    _collect_exits(monkeypatch)
    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert "does not contain project snapshot data" in capsys.readouterr().out
    assert not (tmp_path / "dest").exists()


def test_abr_import_reports_invalid_metadata_not_old_version(monkeypatch, tmp_path):
    state = _make_state()
    _stub_prompt(monkeypatch, str(tmp_path / "dest"))
    monkeypatch.setattr(ask, "ask_file_path", lambda **_kwargs: str(tmp_path / "book.abr.flac"))
    monkeypatch.setattr(project_new_menu_module.AppMetadata, "load_from_file", lambda _path: None)
    monkeypatch.setattr(ProjectTransferUtil, "load_raw_abr_metadata_string", lambda _path: "{bad json")
    errors, _ = _collect_exits(monkeypatch)
    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert errors[0].startswith("Invalid ABR metadata:")
    assert not (tmp_path / "dest").exists()


def test_abr_import_does_not_copy_files_from_cwd_without_source(monkeypatch, tmp_path, capsys):
    state = _make_state()
    _stub_abr_snapshot(monkeypatch, tmp_path, {
        "version": 2, "fish_s2_voice_file_name": ["narrator.flac"],
    })
    monkeypatch.chdir(tmp_path)
    (tmp_path / "narrator.flac").write_bytes(b"unrelated")
    _collect_exits(monkeypatch)
    with _quiet_project_setter():
        assert ProjectNewMenu.make_new_project_using_abr(state) is True
    assert not (tmp_path / "dest" / "narrator.flac").exists()
    assert not (tmp_path / "dest" / PROJECT_VOICE_SUBDIR / "narrator.flac").exists()
    assert "No source project directory was found" in capsys.readouterr().out


def test_abr_import_copy_failure_keeps_previous_project_selected(monkeypatch, tmp_path):
    state = _make_state()
    previous = state.project
    source = tmp_path / "source"
    (source / PROJECT_VOICE_SUBDIR).mkdir(parents=True)
    (source / PROJECT_VOICE_SUBDIR / "narrator.flac").write_bytes(b"voice")
    _stub_abr_snapshot(monkeypatch, tmp_path, {
        "version": 2, "dir_path": str(source), "fish_s2_voice_file_name": ["narrator.flac"],
    })
    errors, _ = _collect_exits(monkeypatch)
    monkeypatch.setattr("tts_audiobook_tool.project_support.project_transfer_util.shutil.copy",
                        lambda *_args: (_ for _ in ()).throw(OSError("copy failed")))
    with _quiet_project_setter():
        assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert "Could not copy supporting file" in errors[0]
    assert state.project is previous and state.prefs.project_dir == ""


def test_abr_import_save_failure_keeps_previous_project_selected(monkeypatch, tmp_path):
    state = _make_state()
    previous = state.project
    _stub_abr_snapshot(monkeypatch, tmp_path, {"version": 2, "max_words": 60})
    errors, _ = _collect_exits(monkeypatch)
    monkeypatch.setattr(Project, "save", lambda self, **_kwargs: "disk full")
    with _quiet_project_setter():
        assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert "disk full" in errors[0]
    assert "Partial files may remain" in errors[0]
    assert state.project is previous and state.prefs.project_dir == ""


def test_abr_import_prefs_failure_keeps_previous_project_selected(monkeypatch, tmp_path):
    state = _make_state()
    previous = state.project
    _stub_abr_snapshot(monkeypatch, tmp_path, {"version": 2, "max_words": 60})
    errors, _ = _collect_exits(monkeypatch)
    monkeypatch.setattr(state.prefs, "save", lambda: "prefs disk full")
    with _quiet_project_setter():
        assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert "prefs disk full" in errors[0]
    assert state.project is previous and state.prefs.project_dir == ""


def test_abr_import_voice_choice_blocks_before_destination_creation(monkeypatch, tmp_path):
    state = _make_state()
    _stub_abr_snapshot(monkeypatch, tmp_path, {
        "version": 3,
        "glm_voice_file_name": "glm.flac",
        "glm_voice_transcript": "glm text",
        "mira_voice_file_name": "mira.flac",
    })
    errors, _ = _collect_exits(monkeypatch)
    def choose(_message):
        assert not (tmp_path / "dest").exists()
        return "1"
    monkeypatch.setattr(ask, "ask_input", choose)
    with _quiet_project_setter():
        assert ProjectNewMenu.make_new_project_using_abr(state) is True
    assert not errors
    assert state.project.voice_references == [{"file_name": "glm.flac", "transcript": "glm text"}]
    saved = json.loads((tmp_path / "dest" / "project.json").read_text())
    assert saved["version"] == 4
    assert saved["voice_references"] == state.project.voice_references
    assert not (tmp_path / "dest" / "project.json.pre-v4.bak").exists()


def test_abr_import_cancelled_voice_choice_keeps_destination_and_selection_untouched(monkeypatch, tmp_path):
    state = _make_state()
    previous = state.project
    _stub_abr_snapshot(monkeypatch, tmp_path, {
        "version": 3, "glm_voice_file_name": "glm.flac", "mira_voice_file_name": "mira.flac",
    })
    errors, _ = _collect_exits(monkeypatch)
    monkeypatch.setattr(ask, "ask_input", lambda _message: "0")
    assert ProjectNewMenu.make_new_project_using_abr(state) is False
    assert "cancelled" in errors[0]
    assert not (tmp_path / "dest").exists()
    assert state.project is previous and state.prefs.project_dir == ""


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
