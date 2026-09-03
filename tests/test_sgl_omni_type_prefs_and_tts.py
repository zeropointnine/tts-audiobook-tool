import json

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def restore_tts_type(had_original_type, original_type):
    if had_original_type:
        setattr(Tts, "_type", original_type)
    else:
        delattr(Tts, "_type")


def write_prefs(tmp_path, sgl_omni_type):
    prefs_dict = {
        "stt_variant": "faster_whisper",
        "stt_config": "cpu_int8",
        "sgl_omni_type": sgl_omni_type,
    }
    prefs_path = tmp_path / PREFS_FILE_NAME
    prefs_path.write_text(json.dumps(prefs_dict), encoding="utf-8")
    return prefs_path


def test_load_sgl_omni_type_auto_detect_values(monkeypatch, tmp_path):
    monkeypatch.setattr("tts_audiobook_tool.app_support.app_paths.get_app_user_dir", lambda: str(tmp_path))

    for raw_value in [None, "", TtsModelType.NONE.value.id]:
        write_prefs(tmp_path, raw_value)
        prefs = Prefs.load(save_if_dirty=False)
        assert prefs.sgl_omni_type is None


def test_load_sgl_omni_type_rejects_invalid_or_local_values(monkeypatch, tmp_path):
    monkeypatch.setattr("tts_audiobook_tool.app_support.app_paths.get_app_user_dir", lambda: str(tmp_path))

    for raw_value in ["does-not-exist", TtsModelType.CHATTERBOX.value.id, 7]:
        write_prefs(tmp_path, raw_value)
        prefs = Prefs.load(save_if_dirty=False)
        assert prefs.sgl_omni_type is None


def test_load_sgl_omni_type_accepts_sgl_omni_values(monkeypatch, tmp_path):
    monkeypatch.setattr("tts_audiobook_tool.app_support.app_paths.get_app_user_dir", lambda: str(tmp_path))

    for expected in TtsModelType.get_sgl_omni_items():
        write_prefs(tmp_path, expected.value.id)
        prefs = Prefs.load(save_if_dirty=False)

        assert prefs.sgl_omni_type == expected


def test_load_legacy_moss_server_type_as_auto_and_rewrite(monkeypatch, tmp_path):
    monkeypatch.setattr("tts_audiobook_tool.app_support.app_paths.get_app_user_dir", lambda: str(tmp_path))
    prefs_path = write_prefs(tmp_path, "server_moss")

    prefs = Prefs.load()

    assert prefs.sgl_omni_type is None
    assert json.loads(prefs_path.read_text(encoding="utf-8"))["sgl_omni_type"] == ""


def test_sgl_omni_type_ids_are_unique():
    ids = [item.value.id for item in TtsModelType]
    assert len(ids) == len(set(ids))


def test_moss_server_variants_have_distinct_catalog_metadata():
    delay = TtsModelType.MOSS_DELAY_SERVER.value
    local = TtsModelType.MOSS_LOCAL_SERVER.value

    assert delay.backend_kind is TtsBackendKind.SGL_OMNI
    assert local.backend_kind is TtsBackendKind.SGL_OMNI
    assert delay.default_output_sample_rate == 24_000
    assert local.default_output_sample_rate == 48_000
    assert delay.ui["proper_name"] != local.ui["proper_name"]


def test_find_moss_server_variant_using_model_id():
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id(
        "OpenMOSS-Team/MOSS-TTS-v1.5"
    ) is TtsModelType.MOSS_DELAY_SERVER
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id(
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
    ) is TtsModelType.MOSS_LOCAL_SERVER


def test_qwen3tts_server_is_sgl_omni_and_non_streaming():
    info = TtsModelType.QWEN3TTS_SERVER.value

    assert info.backend_kind == TtsBackendKind.SGL_OMNI
    assert info.sgl_omni_model_id_substring == "qwen"
    assert info.voice_target_attr == "qwen3_voice_file_name"
    assert info.voice_transcript_attr == "qwen3_voice_transcript"
    assert info.batch_size_attr == "qwen3_server_concurrent_requests"
    assert not info.can_stream


def test_find_tts_type_using_sgl_omni_model_id_finds_qwen3tts_server():
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id("Qwen/Qwen3-TTS") == TtsModelType.QWEN3TTS_SERVER


def test_update_tts_type_uses_explicit_sgl_omni_type_without_model_id_probe(monkeypatch):
    original_type = getattr(Tts, "_type", None)
    had_original_type = hasattr(Tts, "_type")
    original_sgl_omni_type = Tts._sgl_omni_type
    original_base_url = SglOmniUtil._base_url
    explicit_type = TtsModelType.get_sgl_omni_items()[0]

    try:
        Tts._type = TtsModelType.NONE
        Tts._sgl_omni_type = explicit_type
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._base_url = "http://example.test"

        def fail_update_model_id():
            raise AssertionError("explicit SGL-Omni type should not auto-detect model id")

        monkeypatch.setattr(SglOmniUtil, "update_model_id", fail_update_model_id)

        Tts.update_tts_type()

        assert Tts.get_type() == explicit_type
    finally:
        restore_tts_type(had_original_type, original_type)
        Tts._sgl_omni_type = original_sgl_omni_type
        SglOmniUtil._base_url = original_base_url


def test_update_tts_type_auto_detects_when_sgl_omni_type_is_none(monkeypatch):
    original_type = getattr(Tts, "_type", None)
    had_original_type = hasattr(Tts, "_type")
    original_sgl_omni_type = Tts._sgl_omni_type
    original_base_url = SglOmniUtil._base_url
    original_model_id = SglOmniUtil._model_id
    sgl_omni_type = TtsModelType.get_sgl_omni_items()[0]
    calls = []

    try:
        Tts._type = TtsModelType.NONE
        Tts._sgl_omni_type = None
        Tts._backend_mode = TtsBackendKind.SGL_OMNI
        SglOmniUtil._base_url = "http://example.test"

        def update_model_id():
            calls.append(True)
            SglOmniUtil._model_id = sgl_omni_type.value.sgl_omni_model_id_substring

        monkeypatch.setattr(SglOmniUtil, "update_model_id", update_model_id)

        Tts.update_tts_type()

        assert calls == [True]
        assert Tts.get_type() == sgl_omni_type
    finally:
        restore_tts_type(had_original_type, original_type)
        Tts._sgl_omni_type = original_sgl_omni_type
        SglOmniUtil._base_url = original_base_url
        SglOmniUtil._model_id = original_model_id
