"""Explicit setting destinations; sample controls remain kind-only TOML."""
import pytest

from catalog_toml_support import read_catalog, write_catalog
from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppMenuControl, load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH
from tts_audiobook_tool.tts_models.sgl_omni_definition import MenuControl, load_definitions


BACKENDS = [
    ("audio_cpp", "omnivoice_audiocpp", load_audio_cpp_definitions),
    ("sgl_omni", "auk_flash_sglomni", load_definitions),
]
CONTROLS = [
    (backend, model_id, loader, index)
    for backend, model_id, loader in BACKENDS
    for index, control in enumerate(next(entry for entry in read_catalog(CATALOG_PATH)["models"]
                                        if entry["id"] == model_id)[backend]["menu"])
    if control.get("kind") != "voice_samples"
]


def menu_fixture(backend, model_id):
    data = read_catalog(CATALOG_PATH)
    entry = next(entry for entry in data["models"] if entry["id"] == model_id)
    return data, entry[backend]["menu"]


@pytest.mark.parametrize("backend,model_id,loader,index", CONTROLS)
@pytest.mark.parametrize("value", ["missing", "", "advanced", "Voice", 123, True, [], {}])
def test_settings_require_a_valid_string_target(tmp_path, backend, model_id, loader, index, value):
    data, menu = menu_fixture(backend, model_id)
    if value == "missing":
        menu[index].pop("target_menu")
    else:
        menu[index]["target_menu"] = value
    with pytest.raises(ValueError, match=rf"menu\[{index}\]\.target_menu"):
        loader(write_catalog(tmp_path / "invalid.toml", data))


@pytest.mark.parametrize("backend,model_id,loader,index", CONTROLS)
@pytest.mark.parametrize("group", ["advanced", ""])
def test_settings_reject_retired_group_field(tmp_path, backend, model_id, loader, index, group):
    data, menu = menu_fixture(backend, model_id)
    menu[index]["group"] = group
    with pytest.raises(ValueError, match="unsupported.*group"):
        loader(write_catalog(tmp_path / "invalid.toml", data))


@pytest.mark.parametrize("backend,model_id,loader", BACKENDS)
@pytest.mark.parametrize("field,value", [
    ("target_menu", "voice"), ("target_menu", "model"),
    ("parameter", "temperature"), ("label", "Samples"), ("group", "advanced"),
])
def test_voice_samples_reject_every_extra_toml_field(tmp_path, backend, model_id, loader, field, value):
    data, menu = menu_fixture(backend, model_id)
    control = next(control for control in menu if control.get("kind") == "voice_samples")
    control[field] = value
    with pytest.raises(ValueError, match=rf"unsupported.*{field}"):
        loader(write_catalog(tmp_path / "invalid.toml", data))


@pytest.mark.parametrize("backend,model_id,loader,index", CONTROLS)
@pytest.mark.parametrize("target", ["model", "voice"])
def test_both_setting_destinations_load_without_changing_controls(
        tmp_path, backend, model_id, loader, index, target):
    data, menu = menu_fixture(backend, model_id)
    menu[index]["target_menu"] = target
    loaded = loader(write_catalog(tmp_path / "valid.toml", data)).models[model_id].menu
    assert loaded[index].target_menu == target
    assert loaded[index].kind == menu[index].get("kind", "parameter")
    assert loaded[index].parameter == menu[index].get("parameter", "")
    assert loaded[index].label == menu[index].get("label", "")


@pytest.mark.parametrize("backend,loader", [(item[0], item[2]) for item in BACKENDS])
def test_shipped_setting_destinations_and_kind_only_samples(backend, loader):
    definitions = loader().models
    entries = [entry for entry in read_catalog(CATALOG_PATH)["models"] if entry.get("backend_kind") == backend]
    for entry in entries:
        for raw, parsed in zip(entry[backend]["menu"], definitions[entry["id"]].menu, strict=True):
            assert not hasattr(parsed, "group")
            if parsed.kind == "voice_samples":
                assert raw == {"kind": "voice_samples"}
                assert parsed.target_menu is None
            else:
                expected = "voice" if (entry["id"] in ("breeze_tts_2_audiocpp", "omnivoice_audiocpp")
                                       and parsed.kind == "voice_instructions") else "model"
                assert raw["target_menu"] == parsed.target_menu == expected
                assert "group" not in raw


@pytest.mark.parametrize("control_type", [AudioCppMenuControl, MenuControl])
def test_menu_control_constructor_requires_second_destination_field(control_type):
    with pytest.raises(TypeError):
        control_type("seed")
    assert control_type("seed", "voice").target_menu == "voice"
    assert control_type("voice_samples", None).target_menu is None
