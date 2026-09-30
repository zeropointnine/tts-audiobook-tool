from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.ask as ask_module
import tts_audiobook_tool.menus.project_menu as project_menu_module
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.constants_hints import HINT_TOLERANCE_FIRST_CLASS
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.project_menu import on_language
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class SaveableProject(SimpleNamespace):
    save_count: int = 0

    def get_tts_model_type(self) -> TtsModelType:
        return TtsModelType.require_by_id("none")

    def save(self) -> None:
        self.save_count += 1


@pytest.mark.parametrize(("entered_code", "expected_code"), [(" EN ", "en"), ("Es", "es")])
def test_language_change_normalizes_and_shows_tolerance_hint(
    monkeypatch, entered_code: str, expected_code: str
) -> None:
    project = SaveableProject(language_code="fr", strictness=Strictness.LOW)
    prefs = object()
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    hint_calls: list[tuple[object, object, bool]] = []

    stub_language_prompt(monkeypatch, entered_code, hint_calls)

    on_language(state, MenuItem("Language", lambda *_: None))

    assert project.language_code == expected_code
    assert project.save_count == 1
    assert hint_calls == [(prefs, HINT_TOLERANCE_FIRST_CLASS, True)]


@pytest.mark.parametrize("entered_code", ["en-US", "EN", "eng", "es-MX", "spa"])
def test_language_change_shows_tolerance_hint_for_first_class_aliases(
    monkeypatch, entered_code: str
) -> None:
    project = SaveableProject(language_code="fr", strictness=Strictness.LOW)
    prefs = object()
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    hint_calls: list[tuple[object, object, bool]] = []

    stub_language_prompt(monkeypatch, entered_code, hint_calls)

    on_language(state, MenuItem("Language", lambda *_: None))

    assert hint_calls == [(prefs, HINT_TOLERANCE_FIRST_CLASS, True)]


@pytest.mark.parametrize("entered_code", ["", "123", "fr"])
def test_language_change_does_not_show_tolerance_hint_when_not_saved_to_en_or_es(
    monkeypatch, entered_code: str
) -> None:
    project = SaveableProject(language_code="de", strictness=Strictness.LOW)
    state = cast(State, SimpleNamespace(project=project, prefs=object()))
    hint_calls: list[tuple[object, object, bool]] = []

    stub_language_prompt(monkeypatch, entered_code, hint_calls)

    on_language(state, MenuItem("Language", lambda *_: None))

    assert hint_calls == []
    if entered_code == "fr":
        assert project.language_code == "fr"
        assert project.save_count == 1
    else:
        assert project.language_code == "de"
        assert project.save_count == 0


def stub_language_prompt(
    monkeypatch,
    entered_code: str,
    hint_calls: list[tuple[object, object, bool]],
) -> None:
    monkeypatch.setattr(project_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None)
    monkeypatch.setattr(project_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_module, "print_feedback", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_module, "ask_input", lambda *args, **kwargs: entered_code)
    monkeypatch.setattr(
        project_menu_module.Whitelist,
        "set_language_code",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        project_menu_module.hints,
        "show_hint_if_necessary",
        lambda prefs, hint, and_prompt=False: hint_calls.append(
            (prefs, hint, and_prompt)
        ),
    )


@pytest.mark.parametrize("remote_mode", [False, True])
@pytest.mark.parametrize("available", [(), (TtsModelType.require_by_id("vibevoice_local"),)])
def test_project_menu_model_selector_is_remote_only(monkeypatch, tmp_path, remote_mode, available):
    from tts_audiobook_tool.project import Project

    state = cast(State, SimpleNamespace(project=Project(dir_path=str(tmp_path))))
    captured = []
    monkeypatch.setattr(project_menu_module.Tts, "is_remote_mode", lambda: remote_mode)
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models", lambda: list(available))
    monkeypatch.setattr(
        project_menu_module.MenuUtil, "menu",
        lambda current, heading, items, **kwargs: captured.extend(items(current)),
    )

    project_menu_module.ProjectMenu.menu(state)

    labels = [item.label(state) if callable(item.label) else item.label for item in captured]
    model_items = [item for item, label in zip(captured, labels) if label.startswith(("TTS model", "Switch TTS model"))]
    language_item = next(item for item, label in zip(captured, labels) if label.startswith("Language code"))
    assert len(model_items) == int(remote_mode)
    options_item = model_items[0] if remote_mode else language_item
    assert options_item.superlabel == "Options"
    assert sum(item.superlabel == "Options" for item in captured) == 1


@pytest.mark.parametrize("available_count", [0, 1, 2, 3])
def test_unselected_project_model_label_requires_selection_with_multiple_types(monkeypatch, available_count):
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.text_util import strip_ansi_codes
    from tts_audiobook_tool.util import COL_DIM, COL_ERROR, make_menu_label

    state = cast(State, SimpleNamespace(project=Project()))
    models = [TtsModelType.require_by_id("chatterbox_audiocpp"), TtsModelType.require_by_id("higgs_v3_audiocpp"),
              TtsModelType.require_by_id("echo_tts_audiocpp")]
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models", lambda: models[:available_count])

    value = project_menu_module.ProjectMenu.make_tts_model_label(state)
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
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models", lambda: models[:available_count])

    label = project_menu_module.ProjectMenu.make_tts_model_menu_label(state)

    assert strip_ansi_codes(label) == expected
    if available_count > 1:
        assert f"{COL_DIM}({available_count} available)" in label
    else:
        assert "available" not in strip_ansi_codes(label)


@pytest.mark.parametrize("raw", ["higgs_v3_audiocpp", "future-model"])
def test_project_model_label_preserves_selected_and_unknown_labels_without_discovery(monkeypatch, raw):
    from tts_audiobook_tool.project import Project

    state = cast(State, SimpleNamespace(project=Project(tts_model_type=raw)))
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models",
                        lambda: pytest.fail("Only explicit None needs the selection hint"))
    expected = (TtsModelType.require_by_id("higgs_v3_audiocpp").value.ui["proper_name"]
                if raw == "higgs_v3_audiocpp" else "Unknown model: future-model")

    assert project_menu_module.ProjectMenu.make_tts_model_label(state) == expected


@pytest.mark.parametrize("selected", [TtsModelType.require_by_id("vibevoice_local"), TtsModelType.require_by_id("auk_sglomni"), TtsModelType.require_by_id("higgs_v3_audiocpp")])
def test_project_model_picker_saves_only_project_selection(monkeypatch, selected):
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.state import PendingTtsModelChange

    project = Project(tts_model_type="mira_local")
    state = cast(State, SimpleNamespace(
        project=project, prefs=object(),
        pending_tts_model_change=PendingTtsModelChange("chatterbox_local", "mira_local"),
    ))
    captured = {}
    saves = []
    bindings = []
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [selected, selected])
    monkeypatch.setattr(project_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Project, "save", lambda current: saves.append(current.tts_model_type) or "")
    monkeypatch.setattr(project_menu_module.Tts, "bind_project", lambda current: bindings.append(current.tts_model_type))
    monkeypatch.setattr(project_menu_module, "print_feedback", lambda *args: None)

    project_menu_module.ProjectMenu.tts_model_menu(state)

    assert captured["values"] == [TtsModelType.require_by_id("none"), selected]
    assert captured["current_value"] is TtsModelType.require_by_id("mira_local")
    assert captured["labels"][0] == "None (unselected)"
    assert captured["default_value"] is None
    assert all("Auto" not in label for label in captured["labels"])
    assert project.tts_model_type == "mira_local"
    captured["on_select"](selected)
    assert state.pending_tts_model_change is None
    assert saves == bindings == [selected.id]
    captured["on_select"](TtsModelType.require_by_id("none"))
    assert project.tts_model_type == "none"


def test_project_model_picker_preserves_unknown_selection_until_explicit_choice(monkeypatch):
    from tts_audiobook_tool.project import Project

    project = Project(tts_model_type="future-model")
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [])
    monkeypatch.setattr(project_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    project_menu_module.ProjectMenu.tts_model_menu(state)

    assert project.tts_model_type == "future-model"
    assert captured["current_value"] is None
    assert project_menu_module.ProjectMenu.make_tts_model_label(state) == "Unknown model: future-model"


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
    monkeypatch.setattr(project_menu_module.RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models",
                        lambda **kwargs: refreshes.append(kwargs) or [model, other])
    monkeypatch.setattr(project_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    project_menu_module.ProjectMenu.tts_model_menu(state)

    expected = model.value.ui["proper_name"]
    if filename:
        expected += f" {COL_DIM}({filename}){COL_DEFAULT}"
    assert captured["labels"] == ["None (unselected)", expected,
                                   f"{other.value.ui['proper_name']} {COL_DIM}(higgs.gguf){COL_DEFAULT}"]
    assert captured["default_value"] is None
    assert refreshes == [{"refresh": True}]


@pytest.mark.parametrize("metadata_case", ["missing_entry", "missing_path", "ambiguous", "sgl_omni"])
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
    candidates = ((model, "server-id"),)
    if metadata_case == "ambiguous":
        candidates += ((model, "another-server-id"),)
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.SGL_OMNI if metadata_case == "sgl_omni" else TtsBackendKind.AUDIO_CPP,
        models=entries, candidates=candidates,
    )
    captured = {}
    monkeypatch.setattr(project_menu_module.RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(project_menu_module.Tts, "get_available_tts_models", lambda **kwargs: [model])
    monkeypatch.setattr(project_menu_module.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    project_menu_module.ProjectMenu.tts_model_menu(cast(State, SimpleNamespace(project=Project())))

    assert captured["labels"] == ["None (unselected)", model.value.ui["proper_name"]]
