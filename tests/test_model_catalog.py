"""The packaged TOML catalog is the only source of built-in model metadata."""
import tomllib
from pathlib import Path
import subprocess
import sys

import pytest

from catalog_toml_support import catalog_to_toml, read_catalog, write_catalog
from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, load_catalog
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def _copy_catalog(tmp_path, mutate):
    data = read_catalog(CATALOG_PATH)
    mutate(data)
    return write_catalog(tmp_path / "catalog.toml", data)


def test_catalog_fixture_writer_roundtrips_shipped_catalog():
    catalog = read_catalog(CATALOG_PATH)
    assert tomllib.loads(catalog_to_toml(catalog)) == catalog


def test_catalog_comments_are_accepted_and_change_full_file_fingerprint(tmp_path):
    original = CATALOG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "commented.toml"
    path.write_text("# fixture-only comment\n" + original, encoding="utf-8")
    builtins, servers, fingerprint = load_catalog(path)
    shipped_builtins, shipped_servers, shipped_fingerprint = load_catalog()
    assert builtins == shipped_builtins
    assert servers == shipped_servers
    assert fingerprint != shipped_fingerprint


def test_invalid_toml_reports_a_catalog_error(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("schema_version = 3\nmodels = [invalid TOML\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Model catalog.*invalid.toml"):
        load_catalog(path)


def test_catalog_provides_every_builtin_handle_in_order():
    builtins, servers, fingerprint = load_catalog()
    assert list(TtsModelType._builtin_specs) == [spec.id for _, spec in builtins]
    assert len(builtins) == 22
    assert len(servers) == 9  # eight built-in servers plus configured Fun-CosyVoice3
    assert len(fingerprint) == 64
    for symbol, spec in builtins:
        handle = getattr(TtsModelType, symbol)
        assert TtsModelType.get_by_id(spec.id) is handle
        assert handle.value == spec
        assert TtsModelType._builtin_specs[spec.id] is handle.value
    assert TtsModelType.FISH_S2.value.local_torch_devices == [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
    assert TtsModelType.FISH_S2.value.substitutions == [("—", ", "), ("─", ", ")]
    assert TtsModelType.AUK_FLASH_SERVER.value.backend_kind is TtsBackendKind.SGL_OMNI
    assert TtsModelType.NONE.value.backend_kind is None
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK-Flash") is TtsModelType.AUK_FLASH_SERVER


def test_catalog_is_available_at_import_before_model_classes(tmp_path):
    code = """from tts_audiobook_tool.tts_models.model_catalog import load_catalog
assert len(load_catalog()[0]) == 22
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
assert FishS2BaseModel.INFO is TtsModelType.FISH_S2.value
assert TtsModelType.NONE.id == 'none'
import launch
"""
    repo = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env={"PYTHONPATH": repo}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutate, match", [
    (lambda data: data.update(schema_version=2), "schema_version.*version 3"),
    (lambda data: data["models"].append(data["models"][0]), "unique, lowercase model ID"),
    (lambda data: data["models"][0]["spec"].update(unknown=True), "unsupported field.*unknown"),
    (lambda data: data["models"][3]["spec"].update(local_torch_devices=["bad"]), "local_torch_devices"),
    (lambda data: data["models"][2].update(symbol="AUK_SERVER"), "unique uppercase handle"),
    (lambda data: data["models"][1].pop("backend_kind"), "backend_kind.*required for real models"),
    (lambda data: data["models"][3].update(behavior={}), "cannot have server behavior"),
])
def test_rejects_invalid_catalog_without_mutating_live_handles(tmp_path, mutate, match):
    before = TtsModelType.AUK_SERVER.value
    path = _copy_catalog(tmp_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_catalog(path)
    assert TtsModelType.AUK_SERVER.value is before
