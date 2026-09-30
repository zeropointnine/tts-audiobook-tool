from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.app_support.remote_tts_discovery import (
    RemoteTtsDiscovery, RemoteTtsIssue, RemoteTtsSnapshot,
)
from tts_audiobook_tool.menus import options_menu
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.options_menu import OptionsMenu
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_menu_label


@pytest.fixture(autouse=True)
def isolate_remote_discovery(monkeypatch):
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot())
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda *, force=False: RemoteTtsDiscovery.get_snapshot())


@pytest.mark.parametrize("count", [0, 1, 5])
def test_refresh_label_counts_compatible_audio_cpp_models(monkeypatch, count):
    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        models=tuple({"id": f"server-{i}"} for i in range(count + 2)),
        candidates=tuple((model, f"server-{i}") for i in range(count)),
        issue=RemoteTtsIssue("no_supported_models", "No supported models") if count == 0 else None,
    )
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda *a, **kw: pytest.fail("Label must not probe"))
    noun = "model" if count == 1 else "models"
    value = f"{count} compatible {noun} available"

    label = OptionsMenu.make_refresh_remote_tts_label()

    assert label == make_menu_label("Refresh server info", value)
    assert strip_ansi_codes(label) == f"Refresh server info (currently: {value})"


@pytest.mark.parametrize(("count", "noun"), [(0, "models"), (1, "model"), (5, "models")])
def test_refresh_feedback_pluralizes_compatible_models(monkeypatch, count, noun):
    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        candidates=tuple((model, f"server-{i}") for i in range(count)),
    )
    state = cast(State, SimpleNamespace(project=object()))
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda *, force: snapshot)
    monkeypatch.setattr(Tts, "bind_project", lambda project: None)
    feedback = []
    monkeypatch.setattr(options_menu, "print_feedback", lambda *args, **kwargs: feedback.append((args, kwargs)))

    OptionsMenu.refresh_remote_tts(state)

    assert feedback == [(
        ("Remote TTS server refreshed:", f"audio_cpp, {count} compatible {noun}"),
        {"long_pause": True},
    )]


@pytest.mark.parametrize("backend", [None, TtsBackendKind.SGL_OMNI])
def test_refresh_label_unchanged_for_other_or_unknown_backends(monkeypatch, backend):
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(backend_kind=backend))

    assert OptionsMenu.make_refresh_remote_tts_label() == "Refresh server info"


@pytest.mark.parametrize("remote_mode", [False, True])
def test_options_refresh_item_is_remote_only_and_uses_latest_snapshot(monkeypatch, remote_mode):
    state = cast(State, SimpleNamespace(
        project=SimpleNamespace(get_tts_model_type=lambda: TtsModelType.require_by_id("none")),
        prefs=SimpleNamespace(
            aac_bitrate="128k", save_gen_log=False, save_debug_files=False,
            remote_tts_url="http://remote.example:8111", llm_url="",
        ),
    ))
    captured: list[MenuItem] = []
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: remote_mode)
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: True)
    monkeypatch.setattr(MenuUtil, "menu", lambda current, heading, items, **kw: captured.extend(items(current)))

    OptionsMenu.menu(state)

    if not remote_mode:
        labels = [item.label(state) if callable(item.label) else item.label for item in captured]
        assert not any(label.startswith("Refresh server info") for label in labels)
        return

    refresh_item = captured[1]
    assert callable(refresh_item.label)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot())
    assert refresh_item.label(state) == "Refresh server info"
    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        candidates=((model, "server-id"),),
    ))
    assert strip_ansi_codes(refresh_item.label(state)) == (
        "Refresh server info (currently: 1 compatible model available)"
    )


@pytest.mark.parametrize("server_state", ["loaded", "unloaded", "unavailable", "not_configured", "invalid_models"])
def test_remote_options_refreshes_before_rendering_on_each_entry(monkeypatch, server_state):
    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    stale = RemoteTtsSnapshot(backend_kind=TtsBackendKind.SGL_OMNI)
    fresh = (RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        models=({"id": "server-model", "loaded": server_state == "loaded"},),
        candidates=((model, "server-model"),),
    ) if server_state in ("loaded", "unloaded") else RemoteTtsSnapshot(
        issue=RemoteTtsIssue(server_state, "Server discovery failed"),
    ))
    snapshots = [stale]
    events = []
    state = cast(State, SimpleNamespace(
        project=SimpleNamespace(get_tts_model_type=lambda: TtsModelType.require_by_id("none")),
        prefs=SimpleNamespace(
            aac_bitrate="128k", save_gen_log=False, save_debug_files=False,
            remote_tts_url="http://server.test", llm_url="",
        ),
    ))
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: True)
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: True)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: snapshots[0])

    def refresh(*, force):
        assert force is True
        events.append("refresh")
        snapshots[0] = fresh
        return fresh

    def render(current, heading, items, **kwargs):
        assert events[-1] == "refresh"
        assert RemoteTtsDiscovery.get_snapshot() is fresh
        events.append("render")
        controls = items(current)
        labels = [strip_ansi_codes(item.label(current) if callable(item.label) else item.label) for item in controls]
        assert ("Unload audio.cpp models" in labels) == (server_state == "loaded")
        assert callable(controls[0].superlabel)
        if fresh.backend_kind is TtsBackendKind.AUDIO_CPP:
            assert controls[0].superlabel(current) == "Remote TTS server (audio.cpp)"
            assert labels[1] == "Refresh server info (currently: 1 compatible model available)"
        else:
            assert controls[0].superlabel(current) == "Remote TTS server"
            assert labels[1] == "Refresh server info"

    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", refresh)
    monkeypatch.setattr(MenuUtil, "menu", render)

    for _ in range(2):
        snapshots[0] = stale
        OptionsMenu.menu(state)

    assert events == ["refresh", "render", "refresh", "render"]


@pytest.mark.parametrize("remote_mode", [False, True])
def test_options_redraw_does_not_force_another_refresh(monkeypatch, remote_mode):
    from tts_audiobook_tool.menus import menu_util

    render_menu = MenuUtil.menu
    state, items = capture_options_items(
        monkeypatch, remote_mode=remote_mode, model_id="none", use_mlx_whisper=True,
    )
    events = []

    def refresh(*, force=False):
        assert remote_mode, "Local Options must not refresh server discovery"
        assert force is True
        events.append("refresh")
        return RemoteTtsDiscovery.get_snapshot()

    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", refresh)
    monkeypatch.setattr(MenuUtil, "menu", render_menu)
    monkeypatch.setattr(menu_util.MenuStatus, "prepare_tts", lambda state: events.append("prepare"))
    monkeypatch.setattr(menu_util.MenuStatus, "show_pending_tts_model_hint", lambda *a, **kw: None)
    monkeypatch.setattr(MenuUtil, "print_heading", lambda *a, **kw: events.append("heading"))
    monkeypatch.setattr(MenuUtil, "is_first_submenu", False)
    monkeypatch.setattr(menu_util, "printt", lambda *a, **kw: None)
    monkeypatch.setattr(menu_util.ask, "can_hotkey", False)
    llm_index = next(i for i, item in enumerate(items) if callable(item.label) and "LLM settings" in item.label(state))
    hotkeys = iter([str(llm_index + 1), ""])
    monkeypatch.setattr(menu_util.ask, "ask_hotkey", lambda: next(hotkeys))
    monkeypatch.setattr(options_menu.LlmSettingsMenu, "menu", lambda state: events.append("action"))

    OptionsMenu.menu(state)

    assert events == (["refresh"] if remote_mode else []) + [
        "prepare", "heading", "action", "prepare", "heading",
    ]


@pytest.mark.parametrize(("backend", "expected"), [
    (None, "Remote TTS server"),
    (TtsBackendKind.AUDIO_CPP, "Remote TTS server (audio.cpp)"),
    (TtsBackendKind.SGL_OMNI, "Remote TTS server (SGL-Omni)"),
])
def test_remote_heading_uses_cached_backend(monkeypatch, backend, expected):
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(backend_kind=backend))
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda *a, **kw: pytest.fail("Heading must not probe"))

    assert OptionsMenu.make_remote_tts_superlabel() == expected


def capture_options_items(monkeypatch, *, remote_mode, model_id, use_mlx_whisper=False, has_gpu=True):
    import torch

    state = cast(State, SimpleNamespace(
        project=SimpleNamespace(get_tts_model_type=lambda: TtsModelType.require_by_id(model_id)),
        prefs=SimpleNamespace(
            aac_bitrate="128k", save_gen_log=False, save_debug_files=False,
            remote_tts_url="http://127.0.0.1:8080", llm_url="",
            stt_variant=SttVariant.get_default(), stt_config=SttConfig.CUDA_FLOAT16,
            tts_force_cpu=False,
        ),
    ))
    captured: list[MenuItem] = []
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: remote_mode)
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: use_mlx_whisper)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: has_gpu)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(MenuUtil, "menu", lambda current, heading, items, **kw: captured.extend(items(current)))

    OptionsMenu.menu(state)
    return state, captured


@pytest.mark.parametrize("model_id", ["echo_tts_audiocpp", "none", "chatterbox_local"])
@pytest.mark.parametrize("use_mlx_whisper", [False, True])
def test_remote_options_groups_about_before_local_controls(monkeypatch, model_id, use_mlx_whisper):
    state, items = capture_options_items(
        monkeypatch, remote_mode=True, model_id=model_id, use_mlx_whisper=use_mlx_whisper,
    )
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot())
    labels = [strip_ansi_codes(item.label(state) if callable(item.label) else item.label) for item in items]
    has_about = model_id != "none"
    local_start = 3 if has_about else 2
    various_start = local_start + (1 if use_mlx_whisper else 3)

    assert labels[0].startswith("Server URL (currently: ")
    assert labels[1] == "Refresh server info"
    assert callable(items[0].superlabel)
    assert items[0].superlabel(state) == "Remote TTS server"
    assert items[0].superlabel_no_blank_line
    if has_about:
        model_name = state.project.get_tts_model_type().value.ui["proper_name"]
        assert labels[2] == f"About current TTS model: {model_name}"
        called = []
        monkeypatch.setattr(options_menu, "print_about_model", called.append)
        items[2].handler(state, items[2])
        assert called == [state]
    else:
        assert not any("About" in label for label in labels)
    assert not any("Force CPU" in label for label in labels)
    assert items[local_start].superlabel == "Speech-to-text model"
    assert not items[local_start].superlabel_no_blank_line
    if not use_mlx_whisper:
        assert labels[local_start].startswith("Whisper model (currently: ")
        assert labels[local_start + 1].startswith("Whisper device (currently: CUDA, float16")
    assert labels[various_start - 1] == "Unload local models"
    assert items[various_start].superlabel == "Various"
    assert labels[various_start].startswith("LLM settings")
    assert all(not item.hotkey for item in items)


@pytest.mark.parametrize("model_id", ["chatterbox_local", "none"])
@pytest.mark.parametrize("has_gpu", [False, True])
@pytest.mark.parametrize("use_mlx_whisper", [False, True])
def test_local_options_order(monkeypatch, model_id, has_gpu, use_mlx_whisper):
    state, items = capture_options_items(
        monkeypatch, remote_mode=False, model_id=model_id,
        use_mlx_whisper=use_mlx_whisper, has_gpu=has_gpu,
    )
    labels = [strip_ansi_codes(item.label(state) if callable(item.label) else item.label) for item in items]
    model_start = 0 if use_mlx_whisper else 2

    if not use_mlx_whisper:
        assert labels[0].startswith("Whisper model (currently: ")
        assert labels[1].startswith("Whisper device (currently: CUDA, float16")
        assert items[0].superlabel == "Model options"
        assert items[0].superlabel_no_blank_line
    model_labels = []
    if model_id != "none":
        if has_gpu:
            model_labels.append("TTS model - Force CPU (currently: False default)")
        model_labels.append("TTS model - About Chatterbox TTS")
    model_labels.append("Unload local models")
    various_start = model_start + len(model_labels)
    assert labels[model_start:various_start] == model_labels
    assert items[various_start].superlabel == "Various"
    assert all(not item.hotkey for item in items)
    assert not any("Server URL" in label or "Refresh server info" in label for label in labels)


@pytest.mark.parametrize(("remote_mode", "model_id"), [
    (True, "echo_tts_audiocpp"), (True, "none"),
    (False, "chatterbox_local"), (False, "none"),
])
@pytest.mark.parametrize("use_mlx_whisper", [False, True])
def test_options_menu_assigns_sequential_hotkeys_in_display_order(monkeypatch, remote_mode, model_id, use_mlx_whisper):
    from tts_audiobook_tool.menus import menu_util

    render_menu = MenuUtil.menu
    state, items = capture_options_items(
        monkeypatch, remote_mode=remote_mode, model_id=model_id, use_mlx_whisper=use_mlx_whisper,
    )
    assert all(not item.hotkey for item in items)
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot())
    monkeypatch.setattr(menu_util.MenuStatus, "prepare_tts", lambda state: None)
    monkeypatch.setattr(menu_util.MenuStatus, "show_pending_tts_model_hint", lambda *a, **kw: None)
    monkeypatch.setattr(MenuUtil, "print_heading", lambda *a, **kw: None)
    monkeypatch.setattr(MenuUtil, "is_first_submenu", False)
    monkeypatch.setattr(menu_util.ask, "can_hotkey", False)
    monkeypatch.setattr(menu_util.ask, "ask_hotkey", lambda: "")
    output = []
    monkeypatch.setattr(menu_util, "printt", lambda text="": output.append(strip_ansi_codes(text)))

    render_menu(state, "Options", items)

    expected_hotkeys = list("123456789abcdefghijklmnopqrstuvwxyz")[:len(items)]
    assert [item.hotkey for item in items] == expected_hotkeys
    rendered_items = [line for line in output if line.startswith("  [")]
    labels = [strip_ansi_codes(item.label(state) if callable(item.label) else item.label) for item in items]
    assert rendered_items == [
        f"  [{hotkey.upper()}] {label}" for hotkey, label in zip(expected_hotkeys, labels)
    ]


@pytest.mark.parametrize("value", [
    "localhost:bad", "http://[broken", "http://", "ftp://example.test",
    "http://localhost:99999", "http://exa mple.test",
])
def test_malformed_remote_url_edit_does_not_persist_or_bind(monkeypatch, value):
    saved = []
    errors = []
    state = cast(State, SimpleNamespace(
        project=object(),
        prefs=SimpleNamespace(remote_tts_url="http://original.test", save=lambda: saved.append(True)),
    ))
    def ask_url(*, prefill):
        assert prefill == "http://original.test"
        return value

    monkeypatch.setattr(options_menu.AskAdvanced, "ask", ask_url)
    monkeypatch.setattr(options_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(Tts, "bind_project", lambda project: pytest.fail("Invalid edit must not bind"))
    monkeypatch.setattr(options_menu, "print_feedback", lambda *a, **kw: pytest.fail("Invalid edit must not report success"))

    OptionsMenu.ask_remote_tts_url(state)

    assert state.prefs.remote_tts_url == "http://original.test"
    assert saved == []
    assert len(errors) == 1
    assert "Invalid remote TTS server URL" in errors[0]


@pytest.mark.parametrize(("value", "expected"), [
    (" localhost:8000/ ", "http://localhost:8000"),
    ("https://example.test/api/", "https://example.test/api"),
    ("http://[::1]:8000/", "http://[::1]:8000"),
])
def test_valid_remote_url_edit_saves_before_binding(monkeypatch, value, expected):
    events = []
    state = cast(State, SimpleNamespace(
        project=object(),
        prefs=SimpleNamespace(remote_tts_url="http://original.test", save=lambda: events.append("save")),
    ))
    def ask_url(*, prefill):
        assert prefill == "http://original.test"
        return value

    monkeypatch.setattr(options_menu.AskAdvanced, "ask", ask_url)
    monkeypatch.setattr(options_menu.ask, "ask_error", lambda message: pytest.fail(message))
    monkeypatch.setattr(Tts, "bind_project", lambda project: events.append("bind"))
    monkeypatch.setattr(options_menu, "print_feedback", lambda *a, **kw: events.append("feedback"))

    OptionsMenu.ask_remote_tts_url(state)

    assert state.prefs.remote_tts_url == expected
    assert events == ["save", "bind", "feedback"]


@pytest.mark.parametrize("model_id", ["echo_tts_audiocpp", "none"])
@pytest.mark.parametrize("use_mlx_whisper", [False, True])
def test_audio_cpp_unload_uses_all_cached_models_and_follows_about(monkeypatch, model_id, use_mlx_whisper):
    snapshot = RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP,
        models=({"id": "unsupported-model", "loaded": True},),
        issue=RemoteTtsIssue("no_supported_models", "No supported models"),
    )
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: snapshot)
    state, items = capture_options_items(
        monkeypatch, remote_mode=True, model_id=model_id, use_mlx_whisper=use_mlx_whisper,
    )
    labels = [item.label(state) if callable(item.label) else item.label for item in items]
    position = 3 if model_id != "none" else 2

    assert labels[position] == "Unload audio.cpp models"
    assert labels[position - 1].startswith("About current TTS model" if model_id != "none" else "Refresh server info")
    assert items[position + 1].superlabel == "Speech-to-text model"
    assert "Unload local models" in labels
    called = []
    monkeypatch.setattr(OptionsMenu, "unload_audio_cpp_models", lambda: called.append(True))
    items[position].handler(state, items[position])
    assert called == [True]


@pytest.mark.parametrize(("remote_mode", "backend", "models"), [
    (False, TtsBackendKind.AUDIO_CPP, ({"loaded": True},)),
    (True, TtsBackendKind.SGL_OMNI, ({"loaded": True},)),
    (True, None, ({"loaded": True},)),
    (True, TtsBackendKind.AUDIO_CPP, ()),
    (True, TtsBackendKind.AUDIO_CPP, ({"loaded": False},)),
    (True, TtsBackendKind.AUDIO_CPP, ({},)),
    (True, TtsBackendKind.AUDIO_CPP, ({"loaded": None},)),
    (True, TtsBackendKind.AUDIO_CPP, ({"loaded": 1},)),
    (True, TtsBackendKind.AUDIO_CPP, ({"loaded": "true"},)),
])
def test_audio_cpp_unload_hidden_unless_boolean_loaded_in_audio_cpp_mode(monkeypatch, remote_mode, backend, models):
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: RemoteTtsSnapshot(
        backend_kind=backend, models=models,
    ))
    state, items = capture_options_items(monkeypatch, remote_mode=remote_mode, model_id="none")
    labels = [item.label(state) if callable(item.label) else item.label for item in items]
    assert "Unload audio.cpp models" not in labels


@pytest.mark.parametrize("refresh_fails", [False, True])
def test_audio_cpp_unload_success_refreshes_inventory_and_reports_success(monkeypatch, refresh_fails):
    events = []
    snapshots = [RemoteTtsSnapshot(
        backend_kind=TtsBackendKind.AUDIO_CPP, models=({"id": "server-model", "loaded": True},),
    )]
    monkeypatch.setattr(RemoteTtsDiscovery, "get_snapshot", lambda: snapshots[0])
    monkeypatch.setattr(RemoteTtsDiscovery, "get_base_url", lambda: "http://server.test")
    monkeypatch.setattr(options_menu, "print_feedback", lambda message, **kw: events.append((message, kw)))
    monkeypatch.setattr(options_menu.ask, "ask_error", lambda message: pytest.fail(message))
    monkeypatch.setattr(options_menu.AudioCppUtil, "unload_all_models", lambda url: events.append(("unload", url)))

    def refresh(*, force):
        events.append(("refresh", force))
        snapshots[0] = (RemoteTtsSnapshot(issue=RemoteTtsIssue("unavailable", "Connection failed"))
                        if refresh_fails else RemoteTtsSnapshot(
                            backend_kind=TtsBackendKind.AUDIO_CPP,
                            models=({"id": "server-model", "loaded": False},),
                        ))
        return snapshots[0]

    state, items = capture_options_items(monkeypatch, remote_mode=True, model_id="none")
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", refresh)
    unload_item = next(item for item in items if item.label == "Unload audio.cpp models")
    unload_item.handler(state, unload_item)

    assert events == [
        ("Unloading...", {"skip_pause": True}),
        ("unload", "http://server.test"),
        ("refresh", True),
        ("Successfully unloaded models", {}),
    ]
    _, updated_items = capture_options_items(monkeypatch, remote_mode=True, model_id="none")
    assert not any(item.label == "Unload audio.cpp models" for item in updated_items)


def test_audio_cpp_unload_failure_shows_error_without_success_or_refresh(monkeypatch):
    events = []
    monkeypatch.setattr(RemoteTtsDiscovery, "get_base_url", lambda: "http://server.test")
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda **kw: pytest.fail("Failed unload must not refresh"))
    monkeypatch.setattr(options_menu.AudioCppUtil, "unload_all_models", lambda url: "Teardown failed")
    monkeypatch.setattr(options_menu, "print_feedback", lambda message, **kw: events.append((message, kw)))
    monkeypatch.setattr(options_menu.ask, "ask_error", lambda message: events.append(("error", message)))

    OptionsMenu.unload_audio_cpp_models()

    assert events == [("Unloading...", {"skip_pause": True}), ("error", "Teardown failed")]
