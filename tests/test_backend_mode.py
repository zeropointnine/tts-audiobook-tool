"""
Tests for the backend-mode refactor:

- sentinel probe -> immutable process-level backend mode (LOCAL / SGL_OMNI)
- init_local_model_type() mode gating (SGL mode skips the local probe)
- mode-aware requirements-file name for the NONE placeholder
- catalog helpers built on TtsModelSpec.backend_kind
- the longest-substring stopgap in find_tts_type_using_sgl_omni_model_id()
"""

import types
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.moss_server_base_model import MossServerBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

SENTINEL = "tts_audiobook_tool_sgl_omni_marker"


def test_probe_backend_mode_absent_sentinel_is_local():
    # The fence venv (venv-base) does not carry the sentinel
    assert Tts._probe_backend_mode() == TtsBackendKind.LOCAL


def test_probe_backend_mode_present_sentinel_is_sgl_omni(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: types.ModuleType(name) if name == SENTINEL else None,
    )
    assert Tts._probe_backend_mode() == TtsBackendKind.SGL_OMNI


def test_probe_backend_mode_unreadable_sentinel_is_local(monkeypatch):
    def boom(name):
        raise OSError("unreadable")

    monkeypatch.setattr("importlib.util.find_spec", boom)
    assert Tts._probe_backend_mode() == TtsBackendKind.LOCAL


def test_get_backend_mode_probes_lazily_and_caches(monkeypatch):
    calls = []

    def find_spec(name):
        calls.append(name)
        return types.ModuleType(name) if name == SENTINEL else None

    monkeypatch.setattr("importlib.util.find_spec", find_spec)

    Tts._backend_mode = None
    assert Tts.get_backend_mode() == TtsBackendKind.SGL_OMNI
    assert Tts._backend_mode == TtsBackendKind.SGL_OMNI
    Tts.get_backend_mode()
    assert calls.count(SENTINEL) == 1  # probed exactly once


def test_init_local_model_type_in_sgl_mode_skips_local_probe(monkeypatch):
    # Dual-capable venv (sentinel plus a local model library): SGL-Omni wins,
    # the local probe is skipped entirely
    def find_spec(name):
        return types.ModuleType(name) if name in (SENTINEL, "chatterbox") else None

    monkeypatch.setattr("importlib.util.find_spec", find_spec)

    tts_model_type, num_matches = Tts.init_local_model_type()

    assert Tts.get_backend_mode() == TtsBackendKind.SGL_OMNI
    assert tts_model_type == TtsModelType.NONE
    assert num_matches == 0


def test_init_local_model_type_in_local_mode_probes_local_models(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: types.ModuleType(name) if name == "chatterbox" else None,
    )

    tts_model_type, num_matches = Tts.init_local_model_type()

    assert Tts.get_backend_mode() == TtsBackendKind.LOCAL
    assert tts_model_type == TtsModelType.CHATTERBOX
    assert num_matches == 1


def test_start_configures_dots_windows_compile_workaround(monkeypatch):
    from tts_audiobook_tool import start as start_module

    monkeypatch.setattr(
        Tts,
        "init_local_model_type",
        staticmethod(lambda: (TtsModelType.DOTS, 1)),
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
        staticmethod(lambda: (TtsModelType.DOTS, 1)),
    )
    monkeypatch.setattr(start_module.sys, "platform", "win32")
    monkeypatch.setitem(start_module.sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(RuntimeError, match="torch was already imported"):
        object.__new__(start_module.Start).init_tts_or_exit(is_server=False)


def test_get_requirements_file_name_is_mode_aware():
    Tts._type = TtsModelType.NONE
    Tts._backend_mode = TtsBackendKind.LOCAL
    assert Tts.get_requirements_file_name() == "requirements-base.txt"

    Tts._backend_mode = TtsBackendKind.SGL_OMNI
    assert Tts.get_requirements_file_name() == TtsModelType.NONE.value.requirements_file_name

    Tts._type = TtsModelType.CHATTERBOX
    assert Tts.get_requirements_file_name() == TtsModelType.CHATTERBOX.value.requirements_file_name


def test_set_sgl_omni_type_in_local_mode_stores_but_does_not_resolve(monkeypatch):
    original_type = Tts._type
    original_sgl_omni_type = Tts._sgl_omni_type
    original_base_url = SglOmniUtil._base_url
    calls = []

    try:
        Tts._type = TtsModelType.NONE
        Tts._backend_mode = TtsBackendKind.LOCAL
        SglOmniUtil._base_url = "http://example.test"
        monkeypatch.setattr(SglOmniUtil, "update_model_id", lambda: calls.append(True))

        Tts.set_sgl_omni_type(TtsModelType.QWEN3TTS_SERVER)

        assert Tts._sgl_omni_type == TtsModelType.QWEN3TTS_SERVER
        assert Tts.get_type() == TtsModelType.NONE
        assert calls == []
    finally:
        Tts._type = original_type
        Tts._sgl_omni_type = original_sgl_omni_type
        SglOmniUtil._base_url = original_base_url


def _show_startup_hints(monkeypatch, prefs):
    """
    Runs Start.show_startup_hints() against a controlled prefs instance and
    records which hint keys would be shown (both the one-shot and direct
    show paths), without any real prompting.
    """
    from tts_audiobook_tool.app_support import hints as hints_module
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.start import Start

    shown = []
    monkeypatch.setattr(Prefs, "load", staticmethod(lambda save_if_dirty=True: prefs))
    monkeypatch.setattr(
        hints_module, "show_hint_if_necessary",
        lambda p, h, **kw: (shown.append(h.key), True)[1],
    )
    monkeypatch.setattr(
        hints_module, "show_hint",
        lambda h, **kw: (shown.append(h.key), True)[1],
    )

    start = object.__new__(Start)
    start.is_server = False

    start.show_startup_hints()
    return shown


def test_startup_hint_shown_when_sgl_settings_dormant_in_local_mode(monkeypatch):
    # Local mode, no TTS model, but saved SGL-Omni settings: the user's
    # settings are unreachable from this venv, so tell them how to recover
    from tts_audiobook_tool.prefs import Prefs

    shown = _show_startup_hints(
        monkeypatch, Prefs(sgl_omni_type=TtsModelType.QWEN3TTS_SERVER)
    )
    assert "sgl_omni_dormant" in shown


def test_startup_hint_shown_when_only_sgl_url_custom(monkeypatch):
    from tts_audiobook_tool.prefs import Prefs

    shown = _show_startup_hints(
        monkeypatch, Prefs(sgl_omni_url="http://example.test:9009")
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
        Tts._type = TtsModelType.NONE
        Tts._backend_mode = TtsBackendKind.SGL_OMNI

        shown = _show_startup_hints(
            monkeypatch, Prefs(sgl_omni_type=TtsModelType.QWEN3TTS_SERVER)
        )
        assert "sgl_omni_dormant" not in shown
    finally:
        Tts._backend_mode = saved_mode


def test_catalog_helpers_classify_by_backend_kind():
    local_items = TtsModelType.get_local_items()
    sgl_items = TtsModelType.get_sgl_omni_items()

    assert len(local_items) == 14
    assert len(sgl_items) == 5
    assert set(local_items) | set(sgl_items) == set(TtsModelType) - {TtsModelType.NONE}
    assert all(item.value.backend_kind == TtsBackendKind.LOCAL for item in local_items)
    assert all(item.value.backend_kind == TtsBackendKind.SGL_OMNI for item in sgl_items)
    assert TtsModelType.NONE.value.backend_kind is None

    assert TtsModelType.is_backend(TtsModelType.CHATTERBOX, TtsBackendKind.LOCAL)
    assert not TtsModelType.is_backend(TtsModelType.CHATTERBOX, TtsBackendKind.SGL_OMNI)
    assert not TtsModelType.is_valid_sgl_omni_type(TtsModelType.NONE)
    assert not TtsModelType.is_valid_sgl_omni_type(None)
    assert TtsModelType.is_valid_sgl_omni_type(TtsModelType.QWEN3TTS_SERVER)


def test_glm_output_sample_rate_uses_project_value_and_catalog_fallback() -> None:
    project = SimpleNamespace(glm_sr=32_000)
    assert GlmBaseModel.get_output_sample_rate(project) == 32_000

    project.glm_sr = 12_345
    assert GlmBaseModel.get_output_sample_rate(project) == TtsModelType.GLM.value.default_output_sample_rate


def test_moss_output_sample_rate_follows_architecture(monkeypatch) -> None:
    delay_project = SimpleNamespace(moss_target=MossConfigs.DELAY.value.repo_id)
    local_project = SimpleNamespace(moss_target=MossConfigs.LOCAL.value.repo_id)

    assert MossBaseModel.get_output_sample_rate(delay_project) == 24_000
    assert MossBaseModel.get_output_sample_rate(local_project) == 48_000

    monkeypatch.setattr(SglOmniUtil, "get_model_id", lambda: MossConfigs.DELAY.value.repo_id)
    assert MossServerBaseModel.get_output_sample_rate(local_project) == 24_000

    monkeypatch.setattr(SglOmniUtil, "get_model_id", lambda: MossConfigs.LOCAL.value.repo_id)
    assert MossServerBaseModel.get_output_sample_rate(delay_project) == 48_000


def test_only_mira_has_worker_output_filters() -> None:
    assert TtsModelType.MIRA.value.output_filters == ["smem_size"]
    assert all(
        item.value.output_filters == []
        for item in TtsModelType
        if item is not TtsModelType.MIRA
    )


def with_substring(spec, substring):
    # TtsModelSpec is a NamedTuple; _replace is its copy-with-changed-field helper
    return spec._replace(sgl_omni_model_id_substring=substring)


def test_finder_prefers_longest_matching_substring(monkeypatch):
    # Synthetic: a second SGL variant whose substring ("fishs2") contains
    # FISH_S2_SERVER's substring ("fish")
    real_items = TtsModelType.get_sgl_omni_items()
    fake_spec = with_substring(TtsModelType.FISH_S2_SERVER.value, "fishs2")
    fake_item = SimpleNamespace(value=fake_spec)
    monkeypatch.setattr(
        TtsModelType,
        "get_sgl_omni_items",
        staticmethod(lambda: [*real_items, fake_item]),
    )

    # The model id contains both "fish" (real FISH_S2_SERVER) and "fishs2"
    # (the longer synthetic variant) -> the longer match wins
    result = TtsModelType.find_tts_type_using_sgl_omni_model_id("acme/fishs2-v2")
    assert result.value.sgl_omni_model_id_substring == "fishs2"


def test_finder_ties_fall_back_to_catalog_order(monkeypatch):
    # Two variants with the same substring length: the one earlier in the
    # catalog (get_sgl_omni_items() order) wins
    first = TtsModelType.get_sgl_omni_items()[0]
    second = TtsModelType.get_sgl_omni_items()[1]
    shared = "zzzshared"
    monkeypatch.setattr(
        TtsModelType,
        "get_sgl_omni_items",
        staticmethod(lambda: [
            SimpleNamespace(value=with_substring(first.value, shared)),
            SimpleNamespace(value=with_substring(second.value, shared)),
        ]),
    )

    result = TtsModelType.find_tts_type_using_sgl_omni_model_id(f"repo/{shared}")
    assert result.value.sgl_omni_model_id_substring == shared
    assert result.value.id == first.value.id
