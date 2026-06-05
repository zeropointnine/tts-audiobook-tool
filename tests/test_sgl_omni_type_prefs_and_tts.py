import json

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos


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

    for raw_value in [None, "", TtsModelInfos.NONE.value.id]:
        write_prefs(tmp_path, raw_value)
        prefs = Prefs.load(save_if_dirty=False)
        assert prefs.sgl_omni_type is None


def test_load_sgl_omni_type_rejects_invalid_or_local_values(monkeypatch, tmp_path):
    monkeypatch.setattr("tts_audiobook_tool.app_support.app_paths.get_app_user_dir", lambda: str(tmp_path))

    for raw_value in ["does-not-exist", TtsModelInfos.CHATTERBOX.value.id, 7]:
        write_prefs(tmp_path, raw_value)
        prefs = Prefs.load(save_if_dirty=False)
        assert prefs.sgl_omni_type is None


def test_load_sgl_omni_type_accepts_sgl_omni_values(monkeypatch, tmp_path):
    monkeypatch.setattr("tts_audiobook_tool.app_support.app_paths.get_app_user_dir", lambda: str(tmp_path))

    for expected in TtsModelInfos.get_sgl_omni_items():
        write_prefs(tmp_path, expected.value.id)
        prefs = Prefs.load(save_if_dirty=False)

        assert prefs.sgl_omni_type == expected


def test_sgl_omni_type_ids_are_unique():
    ids = [item.value.id for item in TtsModelInfos]
    assert len(ids) == len(set(ids))


def test_update_tts_type_uses_explicit_sgl_omni_type_without_model_id_probe(monkeypatch):
    original_type = getattr(Tts, "_type", None)
    had_original_type = hasattr(Tts, "_type")
    original_sgl_omni_type = Tts._sgl_omni_type
    original_base_url = SglOmniUtil._base_url
    explicit_type = TtsModelInfos.get_sgl_omni_items()[0]

    try:
        Tts._type = TtsModelInfos.NONE
        Tts._sgl_omni_type = explicit_type
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
    sgl_omni_type = TtsModelInfos.get_sgl_omni_items()[0]
    calls = []

    try:
        Tts._type = TtsModelInfos.NONE
        Tts._sgl_omni_type = None
        SglOmniUtil._base_url = "http://example.test"

        def update_model_id():
            calls.append(True)
            SglOmniUtil._model_id = sgl_omni_type.value.server_model_id_substring

        monkeypatch.setattr(SglOmniUtil, "update_model_id", update_model_id)

        Tts.update_tts_type()

        assert calls == [True]
        assert Tts.get_type() == sgl_omni_type
    finally:
        restore_tts_type(had_original_type, original_type)
        Tts._sgl_omni_type = original_sgl_omni_type
        SglOmniUtil._base_url = original_base_url
        SglOmniUtil._model_id = original_model_id
