from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.menus.voice.voice_moss_server_menu import VoiceMossServerMenu
from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.moss_base_model import MossArchType, MossConfigs
from tts_audiobook_tool.tts_models.moss_server_model import (
    MossDelayServerModel,
    MossLocalServerModel,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_moss_server_variants_are_registered_with_distinct_classes():
    assert Tts.get_class_for_type(TtsModelType.MOSS_DELAY_SERVER) is MossDelayServerModel
    assert Tts.get_class_for_type(TtsModelType.MOSS_LOCAL_SERVER) is MossLocalServerModel


def test_moss_server_architecture_hooks_are_fixed_by_variant(monkeypatch):
    def fail_model_id_probe():
        raise AssertionError("explicit MOSS variant must not inspect the server model id")

    monkeypatch.setattr(SglOmniUtil, "get_model_id", fail_model_id_probe)
    monkeypatch.setattr(SglOmniUtil, "update_model_id", fail_model_id_probe)
    project = SimpleNamespace(moss_target=MossConfigs.LOCAL.value.repo_id)

    delay = MossDelayServerModel()
    local = MossLocalServerModel()

    assert delay.get_loaded_arch_type() is MossArchType.DELAY
    assert local.get_loaded_arch_type() is MossArchType.LOCAL
    assert MossDelayServerModel.get_output_sample_rate(project) == 24_000
    assert MossLocalServerModel.get_output_sample_rate(project) == 48_000
    assert not MossDelayServerModel.can_hallucinate_music(project)
    assert MossLocalServerModel.can_hallucinate_music(project)
    assert not MossDelayServerModel.should_trim_trailing_token_noise(project)
    assert MossLocalServerModel.should_trim_trailing_token_noise(project)


@pytest.mark.parametrize(
    ("model_class", "config", "temperature", "top_p", "top_k"),
    [
        (MossDelayServerModel, MossConfigs.DELAY, 1.25, 0.72, 37),
        (MossLocalServerModel, MossConfigs.LOCAL, 0.95, 0.82, 47),
    ],
)
def test_moss_server_generation_uses_variant_project_settings(
        monkeypatch, model_class, config, temperature, top_p, top_k
):
    project = Project()
    project.moss_delay_temperature = 1.25
    project.moss_delay_top_p = 0.72
    project.moss_delay_top_k = 37
    project.moss_local_temperature = 0.95
    project.moss_local_top_p = 0.82
    project.moss_local_top_k = 47
    project.moss_seed = 123

    captured = {}

    def generate_concurrent(base_url, payloads, print_request=False):
        captured["payloads"] = payloads
        return []

    monkeypatch.setattr(SglOmniUtil, "generate_concurrent", generate_concurrent)

    assert model_class().generate_using_project(project, ["Hello"]) == []

    payload = captured["payloads"][0]
    assert payload["temperature"] == temperature
    assert payload["audio_top_p"] == top_p
    assert payload["audio_top_k"] == top_k
    assert payload["seed"] == 123
    assert model_class.CONFIG is config


@pytest.mark.parametrize(
    ("model_class", "config"),
    [
        (MossDelayServerModel, MossConfigs.DELAY),
        (MossLocalServerModel, MossConfigs.LOCAL),
    ],
)
def test_moss_server_generation_uses_variant_defaults(monkeypatch, model_class, config):
    project = Project()
    captured = {}

    def generate_concurrent(base_url, payloads, print_request=False):
        captured["payloads"] = payloads
        return []

    monkeypatch.setattr(SglOmniUtil, "generate_concurrent", generate_concurrent)

    model_class().generate_using_project(project, ["Hello"])

    payload = captured["payloads"][0]
    assert payload["temperature"] == config.value.temperature_default
    assert payload["audio_top_p"] == config.value.audio_top_p_default
    assert payload["audio_top_k"] == config.value.audio_top_k_default


@pytest.mark.parametrize(
    ("model_type", "expected_config"),
    [
        (TtsModelType.MOSS_DELAY_SERVER, MossConfigs.DELAY),
        (TtsModelType.MOSS_LOCAL_SERVER, MossConfigs.LOCAL),
    ],
)
def test_moss_server_menu_derives_config_from_model_type(
        monkeypatch, model_type, expected_config
):
    seen_model_types = []
    seen_configs = []
    state = cast(State, SimpleNamespace(project=Project()))

    monkeypatch.setattr(
        VoiceMossShared,
        "append_voice_items",
        lambda items, actual_state, actual_type: seen_model_types.append(actual_type),
    )
    monkeypatch.setattr(
        VoiceMossShared,
        "make_temperature_item",
        lambda actual_state, config: seen_configs.append(config),
    )
    monkeypatch.setattr(
        VoiceMossShared,
        "make_audio_top_p_item",
        lambda actual_state, config: seen_configs.append(config),
    )
    monkeypatch.setattr(
        VoiceMossShared,
        "make_audio_top_k_item",
        lambda actual_state, config: seen_configs.append(config),
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.menus.voice.voice_moss_server_menu.VoiceMenuShared.make_seed_item",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.menus.voice.voice_moss_server_menu.VoiceMenuShared.menu_wrapper",
        lambda actual_state, make_items: make_items(actual_state),
    )

    VoiceMossServerMenu.menu(state, model_type)

    assert seen_model_types == [model_type]
    assert seen_configs == [expected_config, expected_config, expected_config]


def test_moss_server_menu_rejects_non_server_moss_type():
    with pytest.raises(ValueError, match="Unsupported MOSS server type"):
        state = cast(State, SimpleNamespace(project=Project()))
        VoiceMossServerMenu.menu(state, TtsModelType.MOSS)


def test_moss_menu_attribute_mapping_uses_variant_config():
    assert VoiceMossShared.get_temperature_attr(MossConfigs.DELAY) == "moss_delay_temperature"
    assert VoiceMossShared.get_top_p_attr(MossConfigs.DELAY) == "moss_delay_top_p"
    assert VoiceMossShared.get_top_k_attr(MossConfigs.DELAY) == "moss_delay_top_k"
    assert VoiceMossShared.get_temperature_attr(MossConfigs.LOCAL) == "moss_local_temperature"
    assert VoiceMossShared.get_top_p_attr(MossConfigs.LOCAL) == "moss_local_top_p"
    assert VoiceMossShared.get_top_k_attr(MossConfigs.LOCAL) == "moss_local_top_k"
