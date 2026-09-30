from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def test_sgl_omni_type_ids_are_unique():
    ids = [item.value.id for item in TtsModelType.all()]
    assert len(ids) == len(set(ids))


def test_launcher_uses_registry_handles_without_enum_name():
    import launch

    assert launch.QUALIFIED_MODELS == [
        (member.value.local_module_test, member.value.ui["proper_name"])
        for member in TtsModelType.all()
        if member.id != "none"
    ]


def test_moss_server_variants_have_distinct_catalog_metadata():
    delay = TtsModelType.require_by_id("moss_delay_sglomni").value
    local = TtsModelType.require_by_id("moss_local_sglomni").value

    assert delay.backend_kind is TtsBackendKind.SGL_OMNI
    assert local.backend_kind is TtsBackendKind.SGL_OMNI
    assert delay.default_output_sample_rate == 24_000
    assert local.default_output_sample_rate == 48_000
    assert delay.ui["proper_name"] != local.ui["proper_name"]


def test_find_moss_server_variant_using_model_id():
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id(
        "OpenMOSS-Team/MOSS-TTS-v1.5"
    ) is TtsModelType.require_by_id("moss_delay_sglomni")
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id(
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
    ) is TtsModelType.require_by_id("moss_local_sglomni")


def test_qwen3tts_server_is_sgl_omni_and_non_streaming():
    info = TtsModelType.require_by_id("qwen3tts_sglomni").value

    assert info.backend_kind == TtsBackendKind.SGL_OMNI
    from tts_audiobook_tool.tts_models.sgl_omni_detection import detect_sgl_omni_models
    assert detect_sgl_omni_models([{"id": "Qwen/Qwen3-TTS"}]) == [
        (TtsModelType.require_by_id("qwen3tts_sglomni"), "Qwen/Qwen3-TTS")
    ]
    assert REGISTRY.voice_binding(info.id).group == "qwen3"
    assert REGISTRY.transcript_binding(info.id).group == "qwen3"
    assert REGISTRY.orchestration_binding(info.id).name == "concurrent_requests"
    assert not info.can_stream


def test_find_tts_type_using_sgl_omni_model_id_finds_qwen3tts_server():
    model = TtsModelType.find_tts_type_using_sgl_omni_model_id("Qwen/Qwen3-TTS")
    assert model is not None
    assert model.id == "qwen3tts_sglomni"


def test_sgl_binding_requires_server_to_match_project_selection(monkeypatch):
    selected = TtsModelType.require_by_id("auk_sglomni")
    other = TtsModelType.require_by_id("higgs_v3_sglomni")
    project = Project(tts_model_type=selected.id)
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "http://example.test")
    monkeypatch.setattr(SglOmniUtil, "_model_id", "")
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        TtsBackendKind.SGL_OMNI, candidates=((selected, "org/served-auk"),)))
    assert Tts.bind_project(project) is None
    assert Tts.get_active_type() is selected
    assert Tts._selected_server_model_id == "org/served-auk"
    assert SglOmniUtil._model_id == "org/served-auk"
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        TtsBackendKind.SGL_OMNI, candidates=((other, "org/served-higgs"),)))
    issue = Tts.bind_project(project)
    assert issue is not None and "No server model matches" in issue.verbose
    assert "org/served-higgs" in issue.verbose
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert Tts._selected_server_model_id == ""
    assert project.tts_model_type == selected.id


def test_unselected_project_does_not_auto_follow_sole_sgl_model(monkeypatch):
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.REMOTE_CLIENT)
    monkeypatch.setattr(RemoteTtsDiscovery, "_snapshots", {})
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "http://example.test")
    RemoteTtsDiscovery._store(RemoteTtsSnapshot(
        TtsBackendKind.SGL_OMNI, candidates=((TtsModelType.require_by_id("auk_sglomni"), "org/served-auk"),)))
    assert Tts.get_available_tts_models() == [TtsModelType.require_by_id("auk_sglomni")]
    project = Project()
    assert Tts.bind_project(project) is not None
    assert Tts.get_active_type() is TtsModelType.require_by_id("none")
    assert project.tts_model_type == "none"
