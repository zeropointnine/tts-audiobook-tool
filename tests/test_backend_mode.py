"""
Tests for the backend-mode refactor:

- sentinel probe -> process-level LOCAL / REMOTE_CLIENT runtime mode
- init_local_model_type() skips local probing in remote-client mode
- mode-aware requirements-file name for the NONE placeholder
- catalog helpers built on TtsModelSpec.backend_kind
- SGL-Omni's detector uses catalog-only substring metadata
"""

import types

import pytest

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.project import Project
from project_settings_test_support import set_setting
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

SENTINEL = "tts_audiobook_tool_remote_client_marker"
LEGACY_SENTINEL = "tts_audiobook_tool_sgl_omni_marker"


def test_probe_backend_mode_absent_sentinel_is_local(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert Tts._probe_backend_mode() == TtsRuntimeMode.LOCAL


@pytest.mark.parametrize("markers", [(SENTINEL,), (LEGACY_SENTINEL,), (SENTINEL, LEGACY_SENTINEL)])
def test_probe_backend_mode_present_sentinel_is_remote_client(monkeypatch, markers):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: types.ModuleType(name) if name in markers else None,
    )
    assert Tts._probe_backend_mode() == TtsRuntimeMode.REMOTE_CLIENT


def test_probe_backend_mode_unreadable_sentinel_is_local(monkeypatch):
    def boom(name):
        raise OSError("unreadable")

    monkeypatch.setattr("importlib.util.find_spec", boom)
    assert Tts._probe_backend_mode() == TtsRuntimeMode.LOCAL


@pytest.mark.parametrize("sentinel", [SENTINEL, LEGACY_SENTINEL])
def test_probe_backend_mode_unreadable_other_marker_still_is_remote_client(monkeypatch, sentinel):
    def find_spec(name):
        if name == sentinel:
            return types.ModuleType(name)
        raise OSError("unreadable")

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    assert Tts._probe_backend_mode() == TtsRuntimeMode.REMOTE_CLIENT


@pytest.mark.parametrize("sentinel", [SENTINEL, LEGACY_SENTINEL])
def test_get_backend_mode_probes_lazily_and_caches(monkeypatch, sentinel):
    calls = []

    def find_spec(name):
        calls.append(name)
        return types.ModuleType(name) if name == sentinel else None

    monkeypatch.setattr("importlib.util.find_spec", find_spec)

    Tts._backend_mode = None
    assert Tts.get_backend_mode() == TtsRuntimeMode.REMOTE_CLIENT
    assert Tts._backend_mode == TtsRuntimeMode.REMOTE_CLIENT
    first_calls = list(calls)
    Tts.get_backend_mode()
    assert calls == first_calls  # no additional probes after caching
    assert calls.count(sentinel) == 1


@pytest.mark.parametrize("sentinel", [SENTINEL, LEGACY_SENTINEL])
def test_init_local_model_type_in_remote_mode_skips_local_probe(monkeypatch, sentinel):
    # Dual-capable venv (sentinel plus a local model library): remote-client wins,
    # the local probe is skipped entirely
    def find_spec(name):
        return types.ModuleType(name) if name in (sentinel, "chatterbox") else None

    monkeypatch.setattr("importlib.util.find_spec", find_spec)

    tts_model_type, num_matches = Tts.init_local_model_type()

    assert Tts.get_backend_mode() == TtsRuntimeMode.REMOTE_CLIENT
    assert tts_model_type.id == "none"
    assert num_matches == 0


def test_init_local_model_type_in_local_mode_probes_local_models(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: types.ModuleType(name) if name == "chatterbox" else None,
    )

    tts_model_type, num_matches = Tts.init_local_model_type()

    assert Tts.get_backend_mode() == TtsRuntimeMode.LOCAL
    assert tts_model_type.id == "chatterbox_local"
    assert num_matches == 1


def test_start_configures_dots_windows_compile_workaround(monkeypatch):
    from tts_audiobook_tool import start as start_module

    monkeypatch.setattr(
        Tts,
        "init_local_model_type",
        staticmethod(lambda: (TtsModelType.require_by_id("dots_local"), 1)),
    )
    monkeypatch.setattr(start_module.sys, "platform", "win32")
    monkeypatch.delitem(start_module.sys.modules, "torch", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", raising=False)

    object.__new__(start_module.Start).init_tts_or_exit(is_server=False)

    assert start_module.os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] == "0"


def test_start_rejects_dots_workaround_after_torch_import(monkeypatch):
    from tts_audiobook_tool import start as start_module

    monkeypatch.setattr(
        Tts,
        "init_local_model_type",
        staticmethod(lambda: (TtsModelType.require_by_id("dots_local"), 1)),
    )
    monkeypatch.setattr(start_module.sys, "platform", "win32")
    monkeypatch.setitem(start_module.sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(RuntimeError, match="torch was already imported"):
        object.__new__(start_module.Start).init_tts_or_exit(is_server=False)


def test_get_requirements_file_name_is_mode_aware():
    Tts._type = TtsModelType.require_by_id("none")
    Tts._backend_mode = TtsRuntimeMode.LOCAL
    assert Tts.get_requirements_file_name() == "requirements-base.txt"

    Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT
    assert Tts.get_requirements_file_name() == TtsModelType.require_by_id("none").value.requirements_file_name

    Tts._available_local_models = (TtsModelType.require_by_id("chatterbox_local"),)
    Tts._backend_mode = TtsRuntimeMode.LOCAL
    assert Tts.get_requirements_file_name() == TtsModelType.require_by_id("chatterbox_local").value.requirements_file_name


def _show_startup_hints(monkeypatch, prefs):
    """
    Runs app_hint_util.show_startup_hints() against a controlled prefs
    instance and records which hint keys would be shown (both the one-shot
    and direct show paths), without any real prompting.
    """
    from tts_audiobook_tool.app_support import app_hint_util
    from tts_audiobook_tool.app_support import hints as hints_module

    shown = []
    monkeypatch.setattr(
        hints_module, "show_hint_if_necessary",
        lambda p, h, **kw: (shown.append(h.key), True)[1],
    )
    monkeypatch.setattr(
        hints_module, "show_hint",
        lambda h, **kw: (shown.append(h.key), True)[1],
    )

    app_hint_util.show_shared_startup_hints(prefs, is_server=False)
    return shown


def test_startup_hint_shown_when_sgl_settings_dormant_in_local_mode(monkeypatch):
    # Local mode, no TTS model, but saved SGL-Omni settings: the user's
    # settings are unreachable from this venv, so tell them how to recover
    from tts_audiobook_tool.prefs import Prefs

    shown = _show_startup_hints(
        monkeypatch, Prefs(remote_tts_url="http://example.test")
    )
    assert "sgl_omni_dormant" in shown


def test_startup_hint_shown_when_only_sgl_url_custom(monkeypatch):
    from tts_audiobook_tool.prefs import Prefs

    shown = _show_startup_hints(
        monkeypatch, Prefs(remote_tts_url="http://example.test:9009")
    )
    assert "sgl_omni_dormant" in shown


def test_startup_hint_not_shown_without_sgl_settings(monkeypatch):
    from tts_audiobook_tool.prefs import Prefs

    shown = _show_startup_hints(monkeypatch, Prefs())
    assert "sgl_omni_dormant" not in shown


def test_startup_hint_not_shown_in_sgl_mode(monkeypatch):
    # In SGL-Omni mode the settings are live, not dormant
    from tts_audiobook_tool.prefs import Prefs

    saved_mode = Tts._backend_mode
    try:
        Tts._type = TtsModelType.require_by_id("none")
        Tts._backend_mode = TtsRuntimeMode.REMOTE_CLIENT

        shown = _show_startup_hints(
            monkeypatch, Prefs(remote_tts_url="http://example.test")
        )
        assert "sgl_omni_dormant" not in shown
    finally:
        Tts._backend_mode = saved_mode


def test_catalog_helpers_classify_by_backend_kind():
    local_items = TtsModelType.get_local_items()
    sgl_items = TtsModelType.get_sgl_omni_items()

    audio_items = TtsModelType.get_items_by_backend(TtsBackendKind.AUDIO_CPP)
    assert local_items and sgl_items
    assert audio_items == [TtsModelType.require_by_id("breeze_tts_2_audiocpp"), TtsModelType.require_by_id("chatterbox_audiocpp"),
                           TtsModelType.require_by_id("echo_tts_audiocpp"),
                           # TtsModelType.require_by_id("glm_tts_audiocpp"),  # DISABLED; see model_catalog.toml
                           TtsModelType.require_by_id("higgs_v3_audiocpp"),
                           TtsModelType.require_by_id("moss_delay_audiocpp"),
                           TtsModelType.require_by_id("moss_local_audiocpp"),
                           TtsModelType.require_by_id("omnivoice_audiocpp")]
    assert set(local_items) | set(sgl_items) | set(audio_items) == set(TtsModelType) - {TtsModelType.require_by_id("none")}
    assert all(item.value.backend_kind == TtsBackendKind.LOCAL for item in local_items)
    assert all(item.value.backend_kind == TtsBackendKind.SGL_OMNI for item in sgl_items)
    assert TtsModelType.require_by_id("none").value.backend_kind is None

    assert TtsModelType.is_backend(TtsModelType.require_by_id("chatterbox_local"), TtsBackendKind.LOCAL)
    assert not TtsModelType.is_backend(TtsModelType.require_by_id("chatterbox_local"), TtsBackendKind.SGL_OMNI)
    assert not TtsModelType.is_valid_sgl_omni_type(TtsModelType.require_by_id("none"))
    assert not TtsModelType.is_valid_sgl_omni_type(None)
    assert TtsModelType.is_valid_sgl_omni_type(TtsModelType.require_by_id("qwen3tts_sglomni"))


def test_glm_output_sample_rate_uses_project_value_and_catalog_fallback() -> None:
    project = Project()
    set_setting(project, "glm_sr", 32_000)
    assert GlmBaseModel.get_output_sample_rate(project) == 32_000

    set_setting(project, "glm_sr", 12_345)
    assert GlmBaseModel.get_output_sample_rate(project) == TtsModelType.require_by_id("glm_local").value.default_output_sample_rate


def test_moss_output_sample_rate_follows_architecture(monkeypatch) -> None:
    delay_project = Project()
    set_setting(delay_project, "moss_target", MossConfigs.DELAY.value.repo_id)
    local_project = Project()
    set_setting(local_project, "moss_target", MossConfigs.LOCAL.value.repo_id)

    assert MossBaseModel.get_output_sample_rate(delay_project) == 24_000
    assert MossBaseModel.get_output_sample_rate(local_project) == 48_000

    from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport
    from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions

    monkeypatch.setattr(
        SglOmniUtil,
        "get_model_id",
        lambda: (_ for _ in ()).throw(AssertionError("fixed server variants must not inspect model id")),
    )
    definitions = load_definitions()
    delay_support = ConfiguredModelSupport(definitions.models["moss_delay_sglomni"])
    local_support = ConfiguredModelSupport(definitions.models["moss_local_sglomni"])
    assert delay_support.get_output_sample_rate(local_project) == 24_000
    assert local_support.get_output_sample_rate(delay_project) == 48_000


def test_worker_output_filters_defined_only_where_expected() -> None:
    # MIRA leaks "smem_size" setup info; QWEN3TTS leaks an "open-end generation"
    # pad-token warning. Every other model variant must keep a clean worker log.
    expected = {
        TtsModelType.require_by_id("mira_local"): ["smem_size"],
        TtsModelType.require_by_id("qwen3tts_local"): ["for open-end generation"],
    }
    for item in TtsModelType:
        assert item.value.output_filters == expected.get(item, [])


def test_sgl_detector_prefers_longest_catalog_fragment(monkeypatch):
    from tts_audiobook_tool.tts_models import sgl_omni_detection

    monkeypatch.setattr(sgl_omni_detection, "_matchers", lambda: (
        ("fish_s2_sglomni", "fish"),
        ("qwen3tts_sglomni", "fishs2"),
    ))
    assert sgl_omni_detection.detect_sgl_omni_models([{"id": "acme/FishS2-v2"}]) == [
        (TtsModelType.require_by_id("qwen3tts_sglomni"), "acme/FishS2-v2")
    ]


def test_sgl_detector_ties_keep_catalog_order(monkeypatch):
    from tts_audiobook_tool.tts_models import sgl_omni_detection

    first, second = TtsModelType.get_sgl_omni_items()[:2]
    monkeypatch.setattr(sgl_omni_detection, "_matchers", lambda: (
        (first.id, "zzzshared"), (second.id, "zzzshared"),
    ))
    assert sgl_omni_detection.detect_sgl_omni_models([{"id": "repo/zzzshared"}]) == [
        (first, "repo/zzzshared")
    ]
