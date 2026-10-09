from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.menus.model.model_select_menu as select_menu_module
from tts_audiobook_tool.menus.model.model_select_menu import ModelSelectMenu
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.mark.parametrize("available_count", [0, 1, 2, 3])
def test_unselected_project_model_label_requires_selection_with_multiple_types(monkeypatch, available_count):
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.text_util import strip_ansi_codes
    from tts_audiobook_tool.util import COL_DIM, COL_ERROR, make_menu_label

    state = cast(State, SimpleNamespace(project=Project()))
    models = [TtsModelType.require_by_id("chatterbox_audiocpp"), TtsModelType.require_by_id("higgs_v3_audiocpp"),
              TtsModelType.require_by_id("echo_tts_audiocpp")]
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models", lambda: models[:available_count])

    value = ModelSelectMenu.make_model_label(state)
    label = make_menu_label("TTS model", value)
    if available_count >= 2:
        assert value == f"{COL_ERROR}requires selection"
        assert strip_ansi_codes(label) == "TTS model (currently: requires selection)"
        assert f"{COL_ERROR}requires selection{COL_DIM})" in label
    else:
        assert value == "None (unselected)"
        assert strip_ansi_codes(label) == "TTS model (currently: None (unselected))"


@pytest.mark.parametrize(
    ("available_count", "expected"),
    [
        (0, "TTS model (currently: None (unselected))"),
        (1, "TTS model (currently: None (unselected))"),
        (2, "Switch TTS model (2 available) (currently: requires selection)"),
        (3, "Switch TTS model (3 available) (currently: requires selection)"),
    ],
)
def test_tts_model_menu_label_shows_available_count_only_when_choosing(
    monkeypatch, available_count, expected
):
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.text_util import strip_ansi_codes
    from tts_audiobook_tool.util import COL_DIM

    state = cast(State, SimpleNamespace(project=Project()))
    models = [TtsModelType.require_by_id("chatterbox_audiocpp"), TtsModelType.require_by_id("higgs_v3_audiocpp"),
              TtsModelType.require_by_id("echo_tts_audiocpp")]
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models", lambda: models[:available_count])

    label = ModelSelectMenu.make_menu_label(state)

    assert strip_ansi_codes(label) == expected
    if available_count > 1:
        assert f"{COL_DIM}({available_count} available)" in label
    else:
        assert "available" not in strip_ansi_codes(label)


@pytest.mark.parametrize("raw", ["higgs_v3_audiocpp", "future-model"])
def test_project_model_label_preserves_selected_and_unknown_labels_without_discovery(monkeypatch, raw):
    from tts_audiobook_tool.project import Project

    state = cast(State, SimpleNamespace(project=Project(tts_model_type=raw)))
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models",
                        lambda: pytest.fail("Only explicit None needs the selection hint"))
    expected = (TtsModelType.require_by_id("higgs_v3_audiocpp").value.ui["proper_name"]
                if raw == "higgs_v3_audiocpp" else "Unknown model: future-model")

    assert ModelSelectMenu.make_model_label(state) == expected


@pytest.mark.parametrize("selected", [TtsModelType.require_by_id("vibevoice_local"), TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("higgs_v3_audiocpp")])
def test_project_model_picker_saves_only_project_selection(monkeypatch, selected):
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.state import PendingTtsModelChange

    project = Project(tts_model_type="mira_local")
    state = cast(State, SimpleNamespace(
        project=project, prefs=object(),
        pending_tts_model_change=PendingTtsModelChange("chatterbox_local", "mira_local"),
        pending_project_load_checks=False,
    ))
    captured = {}
    saves = []
    bindings = []
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [selected, selected])
    monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Project, "save", lambda current: saves.append(current.tts_model_type) or "")
    monkeypatch.setattr(select_menu_module.Tts, "bind_project", lambda current: bindings.append(current.tts_model_type))
    monkeypatch.setattr(select_menu_module, "print_feedback", lambda *args: None)

    ModelSelectMenu.menu(state)

    assert captured["values"] == [TtsModelType.require_by_id("none"), selected]
    assert captured["current_value"] is TtsModelType.require_by_id("mira_local")
    assert captured["labels"][0] == "None (unselected)"
    assert captured["default_value"] is None
    assert all("Auto" not in label for label in captured["labels"])
    assert project.tts_model_type == "mira_local"
    captured["on_select"](selected)
    assert state.pending_tts_model_change is None
    assert state.pending_project_load_checks is True
    assert saves == bindings == [selected.id]
    captured["on_select"](TtsModelType.require_by_id("none"))
    assert project.tts_model_type == "none"
    assert state.pending_project_load_checks is True


def test_project_model_picker_sorts_labels_and_keeps_values_aligned(monkeypatch):
    from tts_audiobook_tool.project import Project

    # Discovery order and duplicate entries must not affect the alphabetical picker.
    models = [TtsModelType.require_by_id(model_id) for model_id in
              ("higgs_v3_audiocpp", "echo_tts_audiocpp", "chatterbox_audiocpp")]
    captured = {}
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models",
                        lambda **kwargs: models + [models[0]])
    monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    ModelSelectMenu.menu(cast(State, SimpleNamespace(project=Project())))

    expected = [models[2], models[1], models[0]]
    assert captured["values"] == [TtsModelType.require_by_id("none")] + expected
    assert captured["labels"] == ["None (unselected)"] + [model.value.ui["proper_name"] for model in expected]


def test_project_model_picker_preserves_unknown_selection_until_explicit_choice(monkeypatch):
    from tts_audiobook_tool.project import Project

    project = Project(tts_model_type="future-model")
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [])
    monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    ModelSelectMenu.menu(state)

    assert project.tts_model_type == "future-model"
    assert captured["current_value"] is None
    assert ModelSelectMenu.make_model_label(state) == "Unknown model: future-model"


@pytest.mark.parametrize(
    ("path", "filename"),
    [
        ("/srv/models/chatterbox-q8_0.gguf", "chatterbox-q8_0.gguf"),
        (r"C:\models\chatterbox-q8_0.gguf", "chatterbox-q8_0.gguf"),
        ("chatterbox-q8_0.gguf", "chatterbox-q8_0.gguf"),
        (None, None),
        (123, None),
        ("", None),
        ("   ", None),
        ("/srv/models/", None),
        (".", None),
        ("..", None),
    ],
)
def test_audio_cpp_model_picker_filename(monkeypatch, path, filename):
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsSnapshot
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
    from tts_audiobook_tool.util import COL_DEFAULT, COL_DIM

    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    other = TtsModelType.require_by_id("higgs_v3_audiocpp")
    state = cast(State, SimpleNamespace(project=Project()))
    captured = {}
    refreshes = []
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        # Order and operator-defined server IDs need not match catalog IDs.
        models=({"id": "operator-higgs", "path": "/srv/higgs.gguf"},
                {"id": "operator-chatterbox", "path": path}),
        candidates=((model, "operator-chatterbox"), (other, "operator-higgs")),
    )
    monkeypatch.setattr(select_menu_module.RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models",
                        lambda **kwargs: refreshes.append(kwargs) or [model, other])
    monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    ModelSelectMenu.menu(state)

    expected = model.value.ui["proper_name"]
    if filename:
        expected += f" {COL_DIM}({filename}){COL_DEFAULT}"
    assert captured["labels"] == ["None (unselected)", expected,
                                   f"{other.value.ui['proper_name']} {COL_DIM}(higgs.gguf){COL_DEFAULT}"]
    assert captured["default_value"] is None
    assert refreshes == [{"refresh": True}]


@pytest.mark.parametrize("metadata_case", ["missing_entry", "missing_path", "duplicate_metadata", "sgl_omni"])
def test_model_picker_omits_unavailable_or_non_audio_cpp_filename(monkeypatch, metadata_case):
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsSnapshot
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind

    model = TtsModelType.require_by_id("auk_sglomni" if metadata_case == "sgl_omni" else "chatterbox_audiocpp")
    entries = ({"id": "server-id", "path": "/srv/model.gguf"},)
    if metadata_case == "missing_entry":
        entries = ()
    elif metadata_case == "missing_path":
        entries = ({"id": "server-id"},)
    elif metadata_case == "duplicate_metadata":
        entries += ({"id": "server-id", "path": "/srv/other.gguf"},)
    candidates = ((model, "server-id"),)
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.SGL_OMNI if metadata_case == "sgl_omni" else TtsBackendKind.AUDIO_CPP,
        models=entries, candidates=candidates,
    )
    captured = {}
    monkeypatch.setattr(select_menu_module.RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [model])
    monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    ModelSelectMenu.menu(cast(State, SimpleNamespace(project=Project())))

    assert captured["labels"] == ["None (unselected)", model.value.ui["proper_name"]]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("missing_first_path", [False, True])
def test_audio_cpp_picker_uses_first_matching_entry_filename(monkeypatch, reverse, missing_first_path):
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsSnapshot
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
    from tts_audiobook_tool.util import COL_DEFAULT, COL_DIM

    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    other = TtsModelType.require_by_id("higgs_v3_audiocpp")
    matching_ids = ("z-model", "a-model") if not reverse else ("a-model", "z-model")
    first, second = matching_ids
    first_entry = {"id": first}
    if not missing_first_path:
        first_entry["path"] = f"/srv/{first}.gguf"
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        models=({"id": "unrelated", "path": "/srv/unrelated.gguf"}, first_entry,
                {"id": second, "path": f"/srv/{second}.gguf"}),
        candidates=((other, "unrelated"), (model, first), (model, second)),
    )
    captured = {}
    monkeypatch.setattr(select_menu_module.RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(select_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [model])
    monkeypatch.setattr(select_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    ModelSelectMenu.menu(cast(State, SimpleNamespace(project=Project())))

    expected = model.value.ui["proper_name"]
    if not missing_first_path:
        expected += f" {COL_DIM}({first}.gguf){COL_DEFAULT}"
    assert captured["labels"] == ["None (unselected)", expected]
