"""The packaged TOML catalog is the only source of built-in model metadata."""
import re
import tomllib
from pathlib import Path
import subprocess
import sys

import pytest

from catalog_toml_support import catalog_to_toml, read_catalog, write_catalog
from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, load_catalog, parse_spec
from tts_audiobook_tool.tts_models.model_spec import TtsModelSpec
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
from tts_audiobook_tool.tts_models.sgl_omni_detection import detect_sgl_omni_models
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def _copy_catalog(tmp_path, mutate):
    data = read_catalog(CATALOG_PATH)
    mutate(data)
    return write_catalog(tmp_path / "catalog.toml", data)


def _entry(data, entry_id):
    """Look an entry up by ID so index-based mutations cannot drift."""
    return next(model for model in data["models"] if model["id"] == entry_id)


def test_catalog_fixture_writer_roundtrips_shipped_catalog():
    catalog = read_catalog(CATALOG_PATH)
    assert tomllib.loads(catalog_to_toml(catalog)) == catalog


@pytest.mark.parametrize("model_id", ["chatterbox_local", "echo_tts_audiocpp", "auk_sglomni"])
@pytest.mark.parametrize("note", ["", "Server-controlled settings.\nEdit the deployment to change them."])
def test_optional_settings_note_is_generic_and_roundtrips(tmp_path, model_id, note):
    path = _copy_catalog(tmp_path, lambda data: _entry(data, model_id)["spec"]["ui"].update(settings_note=note))
    catalog = read_catalog(path)
    assert tomllib.loads(catalog_to_toml(catalog)) == catalog
    specs, _, _ = load_catalog(path)
    assert next(spec for spec in specs if spec.id == model_id).ui["settings_note"] == note


@pytest.mark.parametrize("note", [None, True, 123, [], {}])
def test_settings_note_must_be_a_string(note):
    entry = _entry(read_catalog(CATALOG_PATH), "echo_tts_audiocpp")
    entry["spec"]["ui"]["settings_note"] = note
    with pytest.raises(ValueError, match=r"model echo_tts_audiocpp\.spec\.ui\.settings_note: expected str"):
        parse_spec(entry)


def test_settings_note_can_be_omitted():
    entry = _entry(read_catalog(CATALOG_PATH), "chatterbox_audiocpp")
    del entry["spec"]["ui"]["settings_note"]
    assert "settings_note" not in parse_spec(entry).ui


@pytest.mark.parametrize("model_id", ["dots_local", "echo_tts_audiocpp", "auk_sglomni"])
@pytest.mark.parametrize("duration", [5, 15.5])
def test_optional_voice_sample_max_duration_is_generic_and_roundtrips(tmp_path, model_id, duration):
    path = _copy_catalog(tmp_path, lambda data: _entry(data, model_id)["spec"]["ui"].update(
        voice_sample_max_duration_s=duration,
    ))
    specs, _, _ = load_catalog(path)
    assert next(spec for spec in specs if spec.id == model_id).ui["voice_sample_max_duration_s"] == duration


@pytest.mark.parametrize("duration", [None, True, "15", 0, -1, float("inf"), float("nan")])
def test_voice_sample_max_duration_must_be_positive_and_finite(duration):
    entry = _entry(read_catalog(CATALOG_PATH), "echo_tts_audiocpp")
    entry["spec"]["ui"]["voice_sample_max_duration_s"] = duration
    with pytest.raises(ValueError, match=r"spec\.ui\.voice_sample_max_duration_s: expected a positive finite number"):
        parse_spec(entry)


def test_shipped_voice_sample_duration_recommendations():
    specs, _, _ = load_catalog()
    assert {spec.id: spec.ui["voice_sample_max_duration_s"]
            for spec in specs if "voice_sample_max_duration_s" in spec.ui} == {
        "dots_local": 10, "fish_s1_local": 10, "fish_s2_local": 30,
        "higgs_v2_local": 15, "mira_local": 8, "omnivoice_local": 15, "pocket_local": 15,
        "auk_sglomni": 5, "auk_flash_sglomni": 5, "cosyvoice3_sglomni": 30,
        "fish_s2_sglomni": 30, "zonos2_sglomni": 20,
        "cosyvoice3_audiocpp": 30, "dots_audiocpp": 10, "echo_tts_audiocpp": 15, "fish_s2_audiocpp": 30,
        "indextts2_audiocpp": 15, "omnivoice_audiocpp": 15,
    }


def test_catalog_random_seed_cap_defaults_and_int32_limits():
    raw = read_catalog(CATALOG_PATH)
    specs, _, _ = load_catalog()
    assert {entry["id"]: entry["spec"]["max_random_seed"]
            for entry in raw["models"] if "max_random_seed" in entry["spec"]} == {
        "echo_tts_audiocpp": 2147483647,
        "moss_delay_audiocpp": 2147483647,
    }
    for spec in specs:
        expected = 2147483647 if spec.id in ("echo_tts_audiocpp", "moss_delay_audiocpp") else -1
        assert type(spec.max_random_seed) is int
        assert spec.max_random_seed == expected
        assert TtsModelType.require_by_id(spec.id).value.max_random_seed == expected


def test_model_spec_preserves_legacy_positional_constructor():
    spec = TtsModelType.require_by_id("echo_tts_audiocpp").value
    assert TtsModelSpec._fields[-1] == "max_random_seed"
    legacy = TtsModelSpec(*spec[:-1])
    assert legacy[:-1] == spec[:-1]
    assert legacy.max_random_seed == -1


@pytest.mark.parametrize("model_id", ["none", "chatterbox_local", "echo_tts_audiocpp", "auk_sglomni"])
@pytest.mark.parametrize("cap", [-1, 0, 123, 2147483647])
def test_optional_random_seed_cap_is_generic_and_roundtrips(tmp_path, model_id, cap):
    path = _copy_catalog(tmp_path, lambda data: _entry(data, model_id)["spec"].update(max_random_seed=cap))
    catalog = read_catalog(path)
    assert catalog["schema_version"] == 1
    assert tomllib.loads(catalog_to_toml(catalog)) == catalog
    specs, _, _ = load_catalog(path)
    assert next(spec for spec in specs if spec.id == model_id).max_random_seed == cap


@pytest.mark.parametrize("model_id", ["none", "chatterbox_local", "echo_tts_audiocpp", "auk_sglomni"])
@pytest.mark.parametrize("cap, reason", [(True, "expected int"), ("123", "expected int"),
                                        (1.0, "expected int"), (-2, "expected >= -1")])
def test_rejects_invalid_random_seed_cap(tmp_path, model_id, cap, reason):
    path = _copy_catalog(tmp_path, lambda data: _entry(data, model_id)["spec"].update(max_random_seed=cap))
    with pytest.raises(ValueError, match=rf"model {model_id}\.spec\.max_random_seed: {reason}"):
        load_catalog(path)


def test_explicit_null_random_seed_cap_is_not_an_omitted_cap():
    # TOML cannot express null, but parse_spec also accepts in-memory entries.
    entry = _entry(read_catalog(CATALOG_PATH), "none")
    entry["spec"]["max_random_seed"] = None
    with pytest.raises(ValueError, match=r"model none\.spec\.max_random_seed: expected int"):
        parse_spec(entry)


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
    path.write_text("schema_version = 1\nmodels = [invalid TOML\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Model catalog.*invalid.toml"):
        load_catalog(path)


def test_complete_catalog_order_and_backend_suffix_ids():
    raw = read_catalog(CATALOG_PATH)
    expected_ids = [
        "none",
        "chatterbox_local", "dots_local", "fish_s1_local", "fish_s2_local",
        "glm_local", "higgs_v2_local", "indextts2_local", "mira_local",
        "moss_local", "omnivoice_local", "pocket_local", "qwen3tts_local",
        "vibevoice_local",
        # Alphabetize model names, not full IDs: AuK precedes AuK-Flash.
        "auk_sglomni", "auk_flash_sglomni", "cosyvoice3_sglomni",
        "fish_s2_sglomni", "higgs_v3_sglomni", "moss_delay_sglomni",
        "moss_local_sglomni", "qwen3tts_sglomni", "zonos2_sglomni",
        "breeze_tts_2_audiocpp", "chatterbox_audiocpp", "cosyvoice3_audiocpp",
        "dots_audiocpp", "echo_tts_audiocpp", "fireredtts3_audiocpp", "fish_s2_audiocpp", "glm_tts_audiocpp",
        "higgs_v3_audiocpp", "indextts2_audiocpp",
        "moss_delay_audiocpp", "moss_local_audiocpp", "omnivoice_audiocpp",
    ]
    assert raw["schema_version"] == 1
    assert [entry["id"] for entry in raw["models"]] == expected_ids
    assert [spec.id for spec in load_catalog()[0]] == expected_ids
    assert [handle.id for handle in TtsModelType.all()] == expected_ids
    assert "backend_kind" not in raw["models"][0]
    suffixes = {"local": "local", "sgl_omni": "sglomni", "audio_cpp": "audiocpp"}
    for entry in raw["models"][1:]:
        suffix = suffixes[entry["backend_kind"]]
        assert re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*_" + suffix, entry["id"])
        assert not entry["id"].startswith("server_")


def test_catalog_provides_canonical_lookup_handles_in_order():
    specs, servers, fingerprint = load_catalog()
    raw = read_catalog(CATALOG_PATH)
    ids = [entry["id"] for entry in raw["models"]]
    assert all("symbol" not in entry for entry in raw["models"])
    assert [spec.id for spec in specs] == ids
    assert list(TtsModelType._initial_specs) == ids
    assert [handle.id for handle in TtsModelType.all()] == ids
    assert len(specs) == 36
    assert len(servers) == 22  # nine SGL entries plus the thirteen audio.cpp entries
    assert len(fingerprint) == 64
    for spec in specs:
        handle = TtsModelType.require_by_id(spec.id)
        assert TtsModelType.get_by_id(spec.id) is handle
        assert TtsModelType.require_by_id(spec.id) is handle
        assert handle is TtsModelType._initial_catalog[spec.id]
        assert handle.value == spec
        assert TtsModelType._initial_specs[spec.id] == handle.value
    assert TtsModelType.require_by_id("fish_s2_local").value.local_torch_devices == [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
    assert TtsModelType.require_by_id("fish_s2_local").value.substitutions == [("—", ", "), ("─", ", ")]
    assert TtsModelType.require_by_id("auk_flash_sglomni").value.backend_kind is TtsBackendKind.SGL_OMNI
    assert TtsModelType.require_by_id("chatterbox_audiocpp").value.backend_kind is TtsBackendKind.AUDIO_CPP
    assert TtsModelType.require_by_id("chatterbox_audiocpp").value.requires_voice
    assert not TtsModelType.require_by_id("chatterbox_audiocpp").value.can_stream
    assert TtsModelType.require_by_id("higgs_v3_audiocpp").value.backend_kind is TtsBackendKind.AUDIO_CPP
    assert not TtsModelType.require_by_id("higgs_v3_audiocpp").value.requires_voice
    assert not TtsModelType.require_by_id("higgs_v3_audiocpp").value.can_stream
    # GLM-TTS served by audio.cpp shares local glm's output tag and 24 kHz rate
    # but has no samplerate setting: the port's output rate is fixed.
    assert TtsModelType.require_by_id("glm_tts_audiocpp").value.backend_kind is TtsBackendKind.AUDIO_CPP
    assert TtsModelType.require_by_id("glm_tts_audiocpp").value.requires_voice
    assert not TtsModelType.require_by_id("glm_tts_audiocpp").value.can_stream
    assert TtsModelType.require_by_id("glm_tts_audiocpp").value.file_tag == "glm"
    assert TtsModelType.require_by_id("glm_tts_audiocpp").value.default_output_sample_rate == 24000
    assert TtsModelType.require_by_id("none").value.backend_kind is None
    assert TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK-Flash") is TtsModelType.require_by_id("auk_flash_sglomni")


def test_removed_uppercase_model_attributes_are_not_installed():
    assert not hasattr(TtsModelType, "NONE")
    for handle in TtsModelType.all():
        assert not hasattr(TtsModelType, handle.id.upper())


@pytest.mark.parametrize("unknown_id", ["future_catalog_id", "", "NONE", "CHATTERBOX_LOCAL"])
def test_strict_unknown_lookup_raises_and_tolerant_lookup_returns_canonical_none(unknown_id):
    placeholder = TtsModelType.require_by_id("none")
    assert placeholder is TtsModelType.get_by_id("none")
    assert placeholder is TtsModelType._initial_catalog["none"]
    assert placeholder.id == "none"
    with pytest.raises(ValueError) as error:
        TtsModelType.require_by_id(unknown_id)
    assert repr(unknown_id) in str(error.value)
    assert "unknown" in str(error.value).lower()
    assert TtsModelType.get_by_id(unknown_id) is placeholder
    assert unknown_id not in TtsModelType._catalog


def test_lookups_do_not_construct_handles_load_models_discover_or_mutate_catalog(monkeypatch):
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
    from tts_audiobook_tool.tts import Tts
    from tts_audiobook_tool.tts_models import model_catalog

    catalog = TtsModelType._catalog
    specs = TtsModelType._specs
    before_handles = tuple(catalog.items())
    before_specs = tuple(specs.items())
    initial_catalog = TtsModelType._initial_catalog
    initial_specs = TtsModelType._initial_specs

    def unexpected(*args, **kwargs):
        pytest.fail("Lookup must not construct, install, load, or discover models")

    monkeypatch.setattr(TtsModelType, "__init__", unexpected)
    monkeypatch.setattr(TtsModelType, "install_spec", unexpected)
    monkeypatch.setattr(model_catalog, "load_catalog", unexpected)
    monkeypatch.setattr(Tts, "init_local_model_type", unexpected)
    monkeypatch.setattr(Tts, "get_instance", unexpected)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", unexpected)
    monkeypatch.setattr("importlib.util.find_spec", unexpected)
    for model_id, handle in before_handles:
        assert TtsModelType.require_by_id(model_id) is handle
        assert TtsModelType.get_by_id(model_id) is handle
    with pytest.raises(ValueError, match="future_catalog_id"):
        TtsModelType.require_by_id("future_catalog_id")
    assert TtsModelType.get_by_id("future_catalog_id") is catalog["none"]
    assert TtsModelType._catalog is catalog
    assert TtsModelType._specs is specs
    assert tuple(catalog) == tuple(model_id for model_id, _ in before_handles)
    assert tuple(specs) == tuple(model_id for model_id, _ in before_specs)
    assert all(catalog[model_id] is handle for model_id, handle in before_handles)
    assert all(specs[model_id] is spec for model_id, spec in before_specs)
    assert TtsModelType._initial_catalog is initial_catalog
    assert TtsModelType._initial_specs is initial_specs


@pytest.mark.parametrize("saved_id", ["future_catalog_id", "CHATTERBOX_LOCAL", ""])
def test_tolerant_saved_selection_preserves_original_unknown_string(saved_id):
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil

    project = Project(tts_model_type=saved_id)
    assert TtsModelType.get_by_id(saved_id) is TtsModelType.require_by_id("none")
    assert project.get_tts_model_type() is TtsModelType.require_by_id("none")
    assert project.tts_model_type == saved_id
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["tts_model_type"] == saved_id
    reloaded = Project.model_validate(payload)
    assert reloaded.tts_model_type == saved_id
    assert reloaded.get_tts_model_type() is TtsModelType.require_by_id("none")


@pytest.mark.parametrize("model_id", ["none", "chatterbox_local", "auk_sglomni", "cosyvoice3_sglomni"])
def test_installing_existing_spec_preserves_canonical_identity_and_reset(model_id):
    handle = TtsModelType.require_by_id(model_id)
    original = TtsModelType._initial_specs[model_id]
    handles = tuple(TtsModelType.all())
    overlay = original._replace(default_output_sample_rate=original.default_output_sample_rate + 1)
    try:
        assert TtsModelType.install_spec(overlay) is handle
        assert TtsModelType.install_spec(overlay) is handle
        assert TtsModelType.require_by_id(model_id) is handle
        assert TtsModelType.get_by_id(model_id) is handle
        assert handle.value is overlay
        assert all(current is previous for current, previous in zip(TtsModelType.all(), handles))
        assert len(TtsModelType.all()) == len(handles)
    finally:
        TtsModelType.reset_catalog()
    assert TtsModelType.require_by_id(model_id) is handle
    assert TtsModelType.get_by_id(model_id) is handle
    assert handle.value is original


def test_installing_data_only_spec_uses_lookup_without_attributes_and_reset_removes_it():
    placeholder = TtsModelType.require_by_id("none")
    handles = tuple(TtsModelType.all())
    spec = TtsModelType.require_by_id("cosyvoice3_sglomni").value._replace(id="startup_data_only")
    try:
        handle = TtsModelType.register_spec(spec)
        assert TtsModelType.require_by_id("startup_data_only") is handle
        assert TtsModelType.get_by_id("startup_data_only") is handle
        assert TtsModelType.install_spec(spec) is handle
        assert handle.value is spec
        assert TtsModelType.all()[-1] is handle
        assert not hasattr(TtsModelType, "STARTUP_DATA_ONLY")
        with pytest.raises(ValueError, match="Duplicate model ID: startup_data_only"):
            TtsModelType.register_spec(spec)
    finally:
        TtsModelType.reset_catalog()
    with pytest.raises(ValueError, match="startup_data_only"):
        TtsModelType.require_by_id("startup_data_only")
    assert TtsModelType.get_by_id("startup_data_only") is placeholder
    assert all(TtsModelType.require_by_id(handle.id) is handle for handle in handles)
    assert not hasattr(TtsModelType, "STARTUP_DATA_ONLY")


def test_catalog_is_available_at_import_before_model_classes(tmp_path):
    code = """from tts_audiobook_tool.tts_models.model_catalog import load_catalog
assert len(load_catalog()[0]) == 36
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
assert FishS2BaseModel.INFO is TtsModelType.require_by_id("fish_s2_local").value
assert TtsModelType.require_by_id("none").id == "none"
cosyvoice = TtsModelType.require_by_id('cosyvoice3_sglomni')
assert cosyvoice is not TtsModelType.require_by_id("none")
TtsModelType.reset_catalog()
assert TtsModelType.require_by_id('cosyvoice3_sglomni') is cosyvoice
handles = tuple(TtsModelType.all())
from tts_audiobook_tool.tts import Tts
Tts.init_local_model_type()
Tts.init_local_model_type()
assert all(TtsModelType.require_by_id(handle.id) is handle for handle in handles)
TtsModelType.reset_catalog()
assert all(TtsModelType.require_by_id(handle.id) is handle for handle in handles)
import launch
"""
    repo = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env={"PYTHONPATH": repo}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_time_cosyvoice_handle_identity_survives_catalog_reset():
    handle = TtsModelType.require_by_id("cosyvoice3_sglomni")
    spec = handle.value
    assert handle is not TtsModelType.require_by_id("none")
    assert TtsModelType.require_by_id("cosyvoice3_sglomni") is handle
    TtsModelType.reset_catalog()
    assert TtsModelType.get_by_id("cosyvoice3_sglomni") is handle
    assert TtsModelType.require_by_id("cosyvoice3_sglomni") is handle
    assert handle.value == spec
    assert sum(model is handle for model in TtsModelType.all()) == 1


@pytest.mark.parametrize("model_id", ["chatterbox_local", "auk_sglomni", "cosyvoice3_sglomni"])
def test_canonical_handles_survive_overlay_and_reset(model_id):
    handle = TtsModelType.require_by_id(model_id)
    original = handle.value
    overlay = original._replace(default_output_sample_rate=original.default_output_sample_rate + 1)
    try:
        TtsModelType.overlay_spec(overlay)
        assert TtsModelType.require_by_id(model_id) is handle
        assert TtsModelType.get_by_id(model_id) is handle
        assert handle.value is overlay
    finally:
        TtsModelType.reset_catalog()
    assert TtsModelType.require_by_id(model_id) is handle
    assert handle.value is TtsModelType._initial_specs[model_id]


def test_import_accepts_extra_data_only_model_and_preserves_catalog_order(tmp_path):
    from copy import deepcopy

    data = read_catalog(CATALOG_PATH)
    extra = deepcopy(_entry(data, "cosyvoice3_sglomni"))
    extra["id"] = "server_catalog_only"
    extra["spec"]["file_tag"] = "catalog-only"
    data["models"].insert(0, extra)
    path = write_catalog(tmp_path / "extra-model.toml", data)
    code = """import sys
from pathlib import Path
from tts_audiobook_tool.tts_models import model_catalog
load_catalog = model_catalog.load_catalog
model_catalog.load_catalog = lambda: load_catalog(Path(sys.argv[1]))
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
specs, _, _ = load_catalog(Path(sys.argv[1]))
assert [model.id for model in TtsModelType.all()] == [spec.id for spec in specs]
assert not hasattr(TtsModelType, 'SERVER_CATALOG_ONLY')
handle = TtsModelType.require_by_id('server_catalog_only')
assert handle is TtsModelType.all()[0]
assert handle is TtsModelType.get_by_id('server_catalog_only')
TtsModelType.reset_catalog()
assert handle is TtsModelType.all()[0]
assert TtsModelType.require_by_id('server_catalog_only') is handle
assert not hasattr(TtsModelType, 'SERVER_CATALOG_ONLY')
"""
    repo = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path)], cwd=tmp_path,
                            env={"PYTHONPATH": repo}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("import_target", ["metadata", "registry"])
def test_import_time_missing_hardcoded_model_id_reports_actionable_error(tmp_path, import_target):
    data = read_catalog(CATALOG_PATH)
    data["models"].remove(_entry(data, "fish_s2_local"))
    path = write_catalog(tmp_path / "missing-hardcoded-id.toml", data)
    code = """import sys
from pathlib import Path
if sys.argv[2] == 'registry':
    # Cache the model's INFO before removing its handle to isolate registry lookup.
    from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
    del TtsModelType._catalog['fish_s2_local']
    del TtsModelType._specs['fish_s2_local']
    from tts_audiobook_tool.tts import Tts
else:
    from tts_audiobook_tool.tts_models import model_catalog
    load_catalog = model_catalog.load_catalog
    model_catalog.load_catalog = lambda: load_catalog(Path(sys.argv[1]))
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
    assert TtsModelType.get_by_id('fish_s2_local') is TtsModelType.require_by_id('none')
    from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
"""
    repo = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path), import_target], cwd=tmp_path,
                            env={"PYTHONPATH": repo}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert "fish_s2_local" in result.stderr
    assert "unknown" in result.stderr.lower()
    assert "require_by_id" in result.stderr
    assert "AttributeError" not in result.stderr
    source = "fish_s2_base_model.py" if import_target == "metadata" else "tts.py"
    assert source in result.stderr


@pytest.mark.parametrize("mutate, match", [
    (lambda data: data.update(schema_version=2), "schema_version.*version 1"),
    (lambda data: data["models"].append(_entry(data, "none")), "unique, lowercase model ID"),
    (lambda data: _entry(data, "none")["spec"].update(unknown=True), "unsupported field.*unknown"),
    (lambda data: _entry(data, "chatterbox_local")["spec"].update(local_torch_devices=["bad"]), "local_torch_devices"),
    (lambda data: _entry(data, "auk_sglomni").update(symbol="AUK_SGLOMNI"), "unsupported field.*symbol"),
    (lambda data: _entry(data, "auk_flash_sglomni").update(id="AUK_FLASH_SGLOMNI"), "unique, lowercase model ID"),
    (lambda data: data["models"].remove(_entry(data, "none")), "missing built-in none placeholder"),
    (lambda data: _entry(data, "chatterbox_local").pop("backend_kind"), "backend_kind.*required for real models"),
    (lambda data: _entry(data, "chatterbox_local").update(behavior={}), "unsupported field.*behavior"),
    (lambda data: _entry(data, "omnivoice_audiocpp")["audio_cpp"].update(
        request_options={"audio_chunk_threshold": "999"}), "request_options.*finite number"),
    (lambda data: _entry(data, "omnivoice_audiocpp")["audio_cpp"].update(
        request_options={"AudioChunk": 999.0}), "request_options.*invalid request option name"),
])
def test_rejects_invalid_catalog_without_mutating_live_handles(tmp_path, mutate, match):
    before = TtsModelType.require_by_id("auk_sglomni").value
    path = _copy_catalog(tmp_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_catalog(path)
    assert TtsModelType.require_by_id("auk_sglomni").value is before


def test_v5_backend_groups_and_audio_cpp_match_are_isolated():
    raw = read_catalog(CATALOG_PATH)
    audio = {item["id"]: item for item in raw["models"] if item.get("backend_kind") == "audio_cpp"}
    assert set(audio) == {
        "breeze_tts_2_audiocpp", "chatterbox_audiocpp", "cosyvoice3_audiocpp",
        "dots_audiocpp", "echo_tts_audiocpp", "fireredtts3_audiocpp", "fish_s2_audiocpp", "glm_tts_audiocpp",
        "higgs_v3_audiocpp", "indextts2_audiocpp",
        "moss_delay_audiocpp", "moss_local_audiocpp", "omnivoice_audiocpp",
    }

    chatterbox = audio["chatterbox_audiocpp"]
    assert chatterbox["spec"]["ui"]["proper_name"] == "Chatterbox"
    assert chatterbox["spec"]["ui"]["short_name"] == "Chatterbox"
    assert chatterbox["audio_cpp"]["match"] == {
        "family": "chatterbox", "task": "clon", "mode": "offline",
    }
    assert set(chatterbox["audio_cpp"]["parameters"]) == {
        "temperature", "top_p", "repetition_penalty", "guidance_scale", "exaggeration",
    }
    # audio.cpp forwards a fixed set of top-level request fields, so a
    # family-specific control must be declared for the `options` object.
    assert chatterbox["audio_cpp"]["parameters"]["exaggeration"]["target"] == "options"

    higgs = audio["higgs_v3_audiocpp"]
    assert higgs["audio_cpp"]["match"] == {"family": "higgs_audio_tts", "task": "tts", "mode": "offline"}
    assert higgs["audio_cpp"]["reference_transcript"] is True
    assert {name: item["type"] for name, item in higgs["audio_cpp"]["parameters"].items()} == {
        "temperature": "float", "top_p": "float", "top_k": "int",
    }

    breeze = audio["breeze_tts_2_audiocpp"]
    # Both routes select cloning vs instruction-only TTS from the request.
    assert breeze["audio_cpp"]["match"] == {"family": "breeze_tts", "tasks": ["tts", "clon"], "mode": "offline"}
    assert breeze["audio_cpp"]["reference_transcript"] is True
    assert breeze["audio_cpp"]["voice_required"] is False
    assert not TtsModelType.require_by_id(breeze["id"]).value.requires_voice
    assert {name: (item["type"], item["default"], item.get("target", "top_level"))
            for name, item in breeze["audio_cpp"]["parameters"].items()} == {
        "temperature": ("float", 0.9, "top_level"),
        "top_p": ("float", 1.0, "top_level"),
        "top_k": ("int", 50, "top_level"),
        "guidance_scale": ("float", 1.0, "top_level"),
        "instruct": ("str", "", "options"),
    }
    assert breeze["audio_cpp"]["parameters"]["instruct"]["request_key"] == "instruction"

    assert "sgl_omni" not in chatterbox and "match" not in chatterbox
    assert all("sgl_omni" in item and "audio_cpp" not in item and "match" not in item
               for item in raw["models"] if item.get("backend_kind") == "sgl_omni")
    definitions = load_definitions()
    assert len(definitions.models) == 9
    assert chatterbox["id"] not in definitions.models and higgs["id"] not in definitions.models
    assert TtsModelType.require_by_id(chatterbox["id"]) is TtsModelType.require_by_id("chatterbox_audiocpp")
    assert TtsModelType.require_by_id(higgs["id"]) is TtsModelType.require_by_id("higgs_v3_audiocpp")
    assert TtsModelType.require_by_id(breeze["id"]) is TtsModelType.require_by_id("breeze_tts_2_audiocpp")

    echo = audio["echo_tts_audiocpp"]
    assert echo["audio_cpp"]["match"] == {"family": "echo_tts", "task": "clon", "mode": "offline"}
    # Clone needs no transcript, and one request below the server's chunk size
    # stays on the verified single-chunk path.
    assert "reference_transcript" not in echo["audio_cpp"]
    assert echo["audio_cpp"]["max_words_range_reco"] == [20, 50]
    assert {name: (item["type"], item["default"], item.get("target", "top_level"))
            for name, item in echo["audio_cpp"]["parameters"].items()} == {
        "num_inference_steps": ("int", 40, "top_level"),
        "text_guidance_scale": ("float", 3.0, "options"),
        "speaker_guidance_scale": ("float", 8.0, "options"),
    }
    assert TtsModelType.require_by_id(echo["id"]) is TtsModelType.require_by_id("echo_tts_audiocpp")

    omnivoice = audio["omnivoice_audiocpp"]
    # One family/`tts` route serves auto voice, clone and voice design; this
    # entry drives all three, like local `omnivoice_local`.
    assert omnivoice["audio_cpp"]["match"] == {"family": "omnivoice", "task": "tts", "mode": "offline"}
    assert omnivoice["audio_cpp"]["reference_transcript"] is True
    assert omnivoice["audio_cpp"]["voice_required"] is False
    assert {name: (item["type"], item["default"], item.get("target", "top_level"))
            for name, item in omnivoice["audio_cpp"]["parameters"].items()} == {
        "num_inference_steps": ("int", 32, "top_level"),
        "guidance_scale": ("float", 2.0, "top_level"),
        # An audio.cpp request option, not a forwarded top-level field.
        "speed": ("float", 1.0, "options"),
        # Voice-design instruction; the server reads it from `options`.
        "instruct": ("str", "", "options"),
    }
    assert omnivoice["audio_cpp"]["parameters"]["instruct"]["request_key"] == "instruction"
    # The chunker suppression local OmniVoice hardcodes is pinned here rather
    # than exposed: not a parameter, so not a persisted setting either.
    assert omnivoice["audio_cpp"]["request_options"] == {"audio_chunk_threshold": 999.0}
    assert TtsModelType.require_by_id(omnivoice["id"]) is TtsModelType.require_by_id("omnivoice_audiocpp")

    glm = audio["glm_tts_audiocpp"]
    # Both advertised GLM-TTS routes share one reference-conditioned path, so
    # the entry claims both tokens; audio.cpp advertises whichever one the
    # operator configured. It still needs the reference transcript.
    assert glm["audio_cpp"]["match"] == {
        "family": "glm_tts", "tasks": ["tts", "clon"], "mode": "offline",
    }
    assert glm["audio_cpp"]["reference_transcript"] is True
    assert glm["audio_cpp"]["max_words_range_reco"] == [20, 40]
    assert glm["audio_cpp"]["language_policy"] == "omit"
    assert {name: (item["type"], item["default"], item.get("target", "top_level"))
            for name, item in glm["audio_cpp"]["parameters"].items()} == {
        "temperature": ("float", 1.0, "top_level"),
        "top_p": ("float", 0.8, "top_level"),
        "top_k": ("int", 25, "top_level"),
        "num_inference_steps": ("int", 10, "top_level"),
        "flow_guidance_scale": ("float", 0.7, "options"),
    }
    assert TtsModelType.require_by_id(glm["id"]) is TtsModelType.require_by_id("glm_tts_audiocpp")


@pytest.mark.parametrize("model_id, expected", [
    ("breeze_tts_2_audiocpp", [
        ("voice_samples", None, "", ""),
        ("voice_instructions", "voice", "instruct", ""),
        ("parameter", "model", "temperature", "Temperature"),
        ("parameter", "model", "guidance_scale", "Guidance scale"),
        ("parameter", "model", "top_p", "Top-P"),
        ("parameter", "model", "top_k", "Top-K"),
        ("seed", "model", "", ""),
    ]),
    ("higgs_v3_audiocpp", [
        ("voice_samples", None, "", ""),
        ("parameter", "model", "temperature", "Temperature"),
        ("parameter", "model", "top_p", "Top-P"),
        ("parameter", "model", "top_k", "Top-K"),
        ("seed", "model", "", ""),
    ]),
])
def test_explicit_audio_cpp_menus_use_catalog_control_order_and_labels(model_id, expected):
    from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions

    menu = load_audio_cpp_definitions().models[model_id].menu
    assert [(item.kind, item.target_menu, item.parameter, item.label) for item in menu] == expected


@pytest.mark.parametrize("model_id", [
    "breeze_tts_2_audiocpp", "chatterbox_audiocpp", "echo_tts_audiocpp",
    "higgs_v3_audiocpp", "omnivoice_audiocpp",
])
def test_audio_cpp_menu_is_required(model_id):
    entry = _entry(read_catalog(CATALOG_PATH), model_id)
    del entry["audio_cpp"]["menu"]
    with pytest.raises(ValueError, match=r"audio_cpp\.menu: expected a list"):
        parse_spec(entry)


@pytest.mark.parametrize("entry_id, mutate, match", [
    ("chatterbox_audiocpp", lambda entry: entry["audio_cpp"].update(menu={}),
     "audio_cpp.menu.*expected a list"),
    ("chatterbox_audiocpp", lambda entry: entry["audio_cpp"].update(menu=[]),
     "expected each parameter exactly once"),
    ("chatterbox_audiocpp", lambda entry: entry["audio_cpp"]["menu"][1].pop("label"),
     "label.*expected str"),
    ("chatterbox_audiocpp", lambda entry: entry["audio_cpp"]["menu"][1].update(label=""),
     "label.*must not be empty"),
    ("chatterbox_audiocpp", lambda entry: entry.update(sgl_omni={}),
     "audio_cpp models cannot contain sgl_omni"),
    ("chatterbox_audiocpp", lambda entry: entry.update(match={}),
     "unsupported field.*match"),
    ("chatterbox_audiocpp", lambda entry: entry.update(backend_kind="sgl_omni"),
     "sgl_omni models cannot contain audio_cpp"),
    ("chatterbox_audiocpp", lambda entry: entry["audio_cpp"]["match"].update(family=""),
     "family.*must not be empty"),
    ("chatterbox_audiocpp", lambda entry: entry["audio_cpp"]["match"].pop("task"),
     "tasks.*expected a nonempty array"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["match"].update(tasks=["clon"]),
     "either task or tasks, not both"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["match"].pop("task") and entry["audio_cpp"]["match"].update(tasks=[]),
     "tasks.*expected a nonempty array"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["match"].pop("task") and entry["audio_cpp"]["match"].update(tasks=["clon", "clon"]),
     "duplicate task"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["match"].pop("task") and entry["audio_cpp"]["match"].update(tasks=["clon", 3]),
     "tasks.*expected nonempty strings"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["match"].update(session_options={"chatterbox.multilingual_t3": 3}),
     "session_options.*nonempty string"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["temperature"].update(default=3.0),
     "expected min <= default <= max"),
    ("chatterbox_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["temperature"].update(type="number"),
     "expected float, int or str"),
    ("higgs_v3_audiocpp",
     lambda entry: entry["audio_cpp"].update(reference_transcript="yes"),
     "reference_transcript.*expected a boolean"),
    ("higgs_v3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["top_k"].update(target="somewhere"),
     "target.*expected top_level or options"),
    ("higgs_v3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["top_k"].update(default=200),
     "expected min <= default <= max"),
    ("higgs_v3_audiocpp",
     lambda entry: entry["audio_cpp"].update(menu=[{"kind": "voice_samples"}, {"kind": "seed", "target_menu": "model"},
                                                   {"parameter": "missing", "label": "Missing", "target_menu": "model"}]),
     "unknown parameter"),
    ("echo_tts_audiocpp",
     lambda entry: entry["audio_cpp"].update(max_words_range_reco=[50, 20]),
     "1 <= min <= max"),
    ("echo_tts_audiocpp",
     lambda entry: entry["audio_cpp"].update(max_words_range_reco=[50]),
     "min, max, note"),
    ("echo_tts_audiocpp",
     lambda entry: entry["audio_cpp"].update(max_words_range_reco=[20, "50"]),
     "expected integer word counts"),
    ("echo_tts_audiocpp",
     lambda entry: entry["audio_cpp"].update(max_words_range_reco=[20, 50, 3]),
     "expected a string note"),
    ("omnivoice_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["instruct"].update(min=0),
     "min.*not valid for a string parameter"),
    ("omnivoice_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["instruct"].update(target="top_level"),
     "a string parameter must travel in options"),
    ("omnivoice_audiocpp",
     lambda entry: entry["audio_cpp"].update(menu=[{"kind": "voice_samples"}, {"kind": "seed", "target_menu": "model"},
                                                   {"parameter": "instruct", "label": "Instruct", "target_menu": "model"}]),
     "requires the voice_instructions control"),
    ("omnivoice_audiocpp",
     lambda entry: entry["audio_cpp"].update(menu=[{"kind": "voice_samples"}, {"kind": "seed", "target_menu": "model"},
                                                   {"kind": "voice_instructions", "parameter": "speed", "target_menu": "model"}]),
     "requires a string parameter"),
    # Fixed-choice string parameters (CosyVoice3's "Mode").
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["template_name"].update(default="sft"),
     "default must be one of the choices"),
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["template_name"].update(choices=[]),
     "nonempty array of choices"),
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["template_name"]["choices"].append(
         {"value": "instruct", "label": "Again"}),
     "duplicate choice"),
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["top_k"].update(choices=[{"value": "1", "label": "One"}]),
     "choices.*only supported for string parameters"),
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["menu"][1].update(kind="voice_instructions", label=None) or entry["audio_cpp"]["menu"][1].pop("label"),
     "requires the choice control"),
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["menu"][2].update(kind="choice", label="Instruction"),
     "choice requires a string parameter with choices"),
    # Choices are plain data; conditional behavior lives in AudioCppModelBehavior subclasses.
    ("cosyvoice3_audiocpp",
     lambda entry: entry["audio_cpp"]["parameters"]["template_name"]["choices"][0].update(reference_transcript=False),
     "unsupported.*reference_transcript"),
])
def test_audio_cpp_catalog_rejects_invalid_declarations(tmp_path, entry_id, mutate, match):
    data = read_catalog(CATALOG_PATH)
    entry = next(model for model in data["models"] if model["id"] == entry_id)
    mutate(entry)
    with pytest.raises(ValueError, match=match):
        load_catalog(write_catalog(tmp_path / "invalid.toml", data))


@pytest.mark.parametrize("model_id, prefixes, local_only", [
    ("moss_local", ("delay", "local", "local_v15"), True),
    ("moss_delay_sglomni", ("delay",), False),
    ("moss_local_sglomni", ("local",), False),
])
def test_moss_settings_are_private_and_architecture_scoped(model_id, prefixes, local_only):
    raw = read_catalog(CATALOG_PATH)
    assert raw["setting_groups"] == {
        "auk": ["auk_sglomni", "auk_flash_sglomni"],
        "qwen3": ["qwen3tts_local", "qwen3tts_sglomni"],
    }
    expected = [
        {"name": "file_name", "section": "voice_references", "type": "list[str]", "default": []},
        {"name": "transcript", "section": "voice_references", "type": "list[str]", "default": []},
    ]
    if local_only:
        expected.extend([
            {"name": "target", "section": "parameters", "type": "str", "default": ""},
            {"name": "rolling_cont", "section": "parameters", "type": "int", "default": 0},
        ])
    for prefix in prefixes:
        expected.extend([
            {"name": f"{prefix}_temperature", "section": "parameters", "type": "float", "default": -1.0, "sentinel": -1},
            {"name": f"{prefix}_top_p", "section": "parameters", "type": "float", "default": -1.0, "sentinel": -1},
            {"name": f"{prefix}_top_k", "section": "parameters", "type": "int", "default": -1, "sentinel": -1},
        ])
        if local_only:
            # moss_local owns one seed per preset instead of a shared seed.
            expected.append(
                {"name": f"{prefix}_seed", "section": "parameters", "type": "int", "default": -1, "preserve_default": True})
    expected.append({"name": "batch_size", "section": "orchestration", "type": "int", "default": 1})
    if not local_only:
        expected.append(
            {"name": "seed", "section": "parameters", "type": "int", "default": -1, "preserve_default": True})
    assert _entry(raw, model_id)["settings"] == expected


@pytest.mark.parametrize("mutate, match", [
    (lambda data: _entry(data, "higgs_v3_sglomni").update(settings={}), "settings.*expected an array"),
    (lambda data: _entry(data, "higgs_v3_sglomni")["settings"].append(
        _entry(data, "higgs_v3_sglomni")["settings"][0]), "unique nonempty setting names"),
    (lambda data: _entry(data, "higgs_v3_sglomni")["settings"][0].update(extra=True), "unsupported setting declaration"),
    (lambda data: _entry(data, "higgs_v3_sglomni")["settings"][0].update(default="wrong"), "default.*expected"),
    (lambda data: data["setting_groups"]["qwen3"].append("missing"), "setting_groups.qwen3"),
    (lambda data: next(s for s in _entry(data, "qwen3tts_sglomni")["settings"]
                       if s["name"] == "temperature").update(group="unauthorized"), "unauthorized shared group"),
])
def test_rejects_invalid_setting_declarations_atomically(tmp_path, mutate, match):
    from tts_audiobook_tool.project_support.model_settings import REGISTRY

    before = dict(REGISTRY.bindings)
    with pytest.raises(ValueError, match=match):
        load_catalog(_copy_catalog(tmp_path, mutate))
    assert REGISTRY.bindings == before


@pytest.mark.parametrize("model_id", ["higgs_v3_sglomni", "higgs_v3_audiocpp"])
@pytest.mark.parametrize("field, value, match", [
    ("default", 101, "settings.top_k.default.*within backend bounds"),
    ("sentinel", -2, "settings.top_k.sentinel.*match the backend default_sentinel"),
])
def test_backend_storage_defaults_and_sentinels_match_parameter_contracts(tmp_path, model_id, field, value, match):
    data = read_catalog(CATALOG_PATH)
    entry = _entry(data, model_id)
    setting = next(s for s in entry["settings"] if s["name"] == "top_k")
    setting[field] = value
    with pytest.raises(ValueError, match=match):
        load_catalog(write_catalog(tmp_path / "bad-storage.toml", data))


@pytest.mark.parametrize("name", ["file_name", "transcript"])
def test_voice_and_transcript_cannot_have_different_storage_owners(tmp_path, name):
    data = read_catalog(CATALOG_PATH)
    setting = next(s for s in _entry(data, "qwen3tts_sglomni")["settings"] if s["name"] == name)
    setting.pop("group")
    with pytest.raises(ValueError, match="settings.transcript.*file_name storage owner"):
        load_catalog(write_catalog(tmp_path / "split-voice-owner.toml", data))


def test_missing_shipped_data_only_cosyvoice_definition_is_rejected(tmp_path):
    data = read_catalog(CATALOG_PATH)
    data["models"] = [entry for entry in data["models"] if entry["id"] != "cosyvoice3_sglomni"]
    with pytest.raises(ValueError, match="missing built-in.*cosyvoice3_sglomni"):
        load_definitions(write_catalog(tmp_path / "missing-cosyvoice.toml", data))


def test_sgl_backend_detector_returns_catalog_and_served_id_pairs():
    assert detect_sgl_omni_models([
        {"id": "tencent/AuK-Flash"}, {"id": "OpenMOSS/MOSS-TTS-Local"},
        {"id": "chatterbox/audio_cpp"}, {"id": ""}, {"bad": "x"},
    ]) == [
        (TtsModelType.require_by_id("auk_flash_sglomni"), "tencent/AuK-Flash"),
        (TtsModelType.require_by_id("moss_local_sglomni"), "OpenMOSS/MOSS-TTS-Local"),
    ]
