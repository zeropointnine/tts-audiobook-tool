"""Backend adapters behind the shared catalog settings menus."""
from dataclasses import replace

import pytest

from tts_audiobook_tool.menus.model.catalog_menu_source import (
    AudioCppMenuSource,
    LocalMenuSource,
    SglOmniMenuSource,
    catalog_menu_source,
    get_qwen_speaker_choices,
)
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.local_definition import get_local_definition
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.fixture(scope="module")
def sgl():
    return load_definitions().models


@pytest.fixture(scope="module")
def audio_cpp():
    return load_audio_cpp_definitions().models


def test_factory_selects_adapter_by_definition_kind(sgl, audio_cpp):
    assert isinstance(catalog_menu_source(sgl["higgs_v3_sglomni"]), SglOmniMenuSource)
    assert isinstance(catalog_menu_source(audio_cpp["breeze_tts_2_audiocpp"]), AudioCppMenuSource)
    assert isinstance(catalog_menu_source(get_local_definition("mira_local")), LocalMenuSource)
    with pytest.raises(TypeError):
        catalog_menu_source(object())


@pytest.mark.parametrize("backend,model_id,name,style,hint", [
    # MOSS temperatures use the stock editor plus a hint on both backends.
    ("sgl", "moss_delay_sglomni", "delay_temperature", "standard", True),
    ("sgl", "moss_delay_sglomni", "delay_top_p", "sentinel_reset", False),
    ("sgl", "higgs_v3_sglomni", "temperature", "sentinel_reset", False),
    ("audio_cpp", "moss_delay_audiocpp", "delay_temperature", "standard", True),
    ("audio_cpp", "breeze_tts_2_audiocpp", "temperature", "standard", False),
])
def test_edit_style_and_moss_hint_per_backend(sgl, audio_cpp, backend, model_id, name, style, hint):
    source = catalog_menu_source((sgl if backend == "sgl" else audio_cpp)[model_id])
    parameter = source.numeric(name)
    assert source.edit_style(parameter) == style
    assert source.shows_moss_hint(parameter) is hint


def test_request_cap_only_comes_from_sgl_omni_max_request_value(sgl, audio_cpp):
    fish = catalog_menu_source(sgl["fish_s2_sglomni"])
    assert fish.request_cap(fish.numeric("top_k")) == 30
    assert fish.request_cap(fish.numeric("temperature")) is None
    breeze = catalog_menu_source(audio_cpp["breeze_tts_2_audiocpp"])
    assert breeze.request_cap(breeze.numeric("top_k")) is None


def test_audio_cpp_rejects_mismatched_control_kinds(audio_cpp):
    definition = audio_cpp["cosyvoice3_audiocpp"]
    source = catalog_menu_source(definition)
    with pytest.raises(ValueError, match="template_name is not a numeric parameter"):
        source.numeric("template_name")
    with pytest.raises(ValueError, match="top_k is not a choice parameter"):
        source.choice("top_k")
    # A string parameter without choices cannot back a choice control either.
    no_choices = replace(definition.parameters["template_name"], choices=())
    source = catalog_menu_source(replace(definition, parameters={**definition.parameters, "template_name": no_choices}))
    with pytest.raises(ValueError, match="template_name is not a choice parameter"):
        source.choice("template_name")


def test_only_qwen3_audio_cpp_offers_speaker_presets(sgl, audio_cpp):
    qwen = catalog_menu_source(audio_cpp["qwen3tts_audiocpp"])
    assert qwen.voice_preset_choices(TtsModelType.require_by_id("qwen3tts_audiocpp")) == (
        get_qwen_speaker_choices, True)
    breeze = catalog_menu_source(audio_cpp["breeze_tts_2_audiocpp"])
    assert breeze.voice_preset_choices(TtsModelType.require_by_id("breeze_tts_2_audiocpp")) == (None, False)
    qwen_sgl = catalog_menu_source(sgl["qwen3tts_sglomni"])
    assert qwen_sgl.voice_preset_choices(TtsModelType.require_by_id("qwen3tts_sglomni")) == (None, False)
