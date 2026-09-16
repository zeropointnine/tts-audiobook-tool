from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from tts_audiobook_tool.app_types import ReadinessIssue, Sound
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.menus.voice.voice_auk_server_menu import VoiceAuKServerMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.auk_server_model import (
    AuKBaseServerModel,
    AuKFlashServerModel,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


@pytest.mark.parametrize(
    ("model_type", "model_class", "substring", "proper_name"),
    [
        (TtsModelType.AUK_SERVER, AuKBaseServerModel, "auk", "AuK"),
        (
            TtsModelType.AUK_FLASH_SERVER,
            AuKFlashServerModel,
            "auk-flash",
            "AuK-Flash",
        ),
    ],
)
def test_auk_variants_have_distinct_catalog_and_runtime_identity(
    model_type, model_class, substring, proper_name
):
    info = model_type.value

    assert info.backend_kind is TtsBackendKind.SGL_OMNI
    assert info.sgl_omni_model_id_substring == substring
    assert info.ui["proper_name"] == proper_name
    assert info.default_output_sample_rate == 24_000
    assert info.voice_target_attr == "auk_voice_file_name"
    assert info.voice_transcript_attr == "auk_voice_transcript"
    assert info.batch_size_attr == "auk_server_concurrent_requests"
    assert info.requires_voice
    assert not info.can_stream
    assert info.requirements_file_name == "requirements-sgl-omni.txt"
    assert Tts.get_class_for_type(model_type) is model_class
    assert model_class.INFO is info


def test_find_auk_variant_using_sgl_omni_model_id():
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK")
        is TtsModelType.AUK_SERVER
    )
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id("tencent/AuK-Flash")
        is TtsModelType.AUK_FLASH_SERVER
    )
    assert (
        TtsModelType.find_tts_type_using_sgl_omni_model_id(
            "/models/TENCENT/AUK-FLASH"
        )
        is TtsModelType.AUK_FLASH_SERVER
    )


def test_auk_project_fields_are_shared_normalized_and_serialized():
    project = Project.model_validate(
        {
            "auk_voice_file_name": "voice.flac",
            "auk_voice_transcript": "reference transcript",
            "auk_server_concurrent_requests": 0,
            "auk_speed": "invalid",
            "auk_seed": "invalid",
        }
    )

    assert project.auk_voice_file_name == ["voice.flac"]
    assert project.auk_voice_transcript == ["reference transcript"]
    assert project.auk_server_concurrent_requests == 1
    assert project.auk_speed == 1.0
    assert project.auk_seed == -1

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["auk_voice_file_name"] == "voice.flac"
    assert payload["auk_voice_transcript"] == "reference transcript"
    assert payload["auk_server_concurrent_requests"] == 1
    assert payload["auk_speed"] == 1.0
    assert payload["auk_seed"] == -1


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_auk_non_finite_integer_fields_are_normalized(invalid_value):
    project = Project.model_validate(
        {
            "auk_server_concurrent_requests": invalid_value,
            "auk_seed": invalid_value,
        }
    )

    assert project.auk_server_concurrent_requests == 1
    assert project.auk_seed == -1


def test_auk_boolean_integer_fields_are_normalized():
    project = Project.model_validate(
        {
            "auk_server_concurrent_requests": True,
            "auk_seed": True,
        }
    )

    assert project.auk_server_concurrent_requests == 1
    assert project.auk_seed == -1


@pytest.mark.parametrize("model_class", [AuKBaseServerModel, AuKFlashServerModel])
def test_auk_readiness_requires_voice_and_transcript(monkeypatch, tmp_path, model_class):
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_base_model.SglOmniUtil.check_readiness",
        lambda _: None,
    )

    no_voice_issues = model_class.get_blocking_issues(Project(), None)
    assert any(issue.short == "voice sample" for issue in no_voice_issues)

    voice_path = tmp_path / "voice.flac"
    voice_path.write_bytes(b"audio")
    missing_transcript_issues = model_class.get_blocking_issues(
        Project.model_validate(
            {
                "auk_voice_file_name": str(voice_path),
            }
        ),
        None,
    )
    assert any(
        issue.short == "voice clone transcript"
        for issue in missing_transcript_issues
    )

    blank_transcript_issues = model_class.get_blocking_issues(
        Project.model_validate(
            {
                "auk_voice_file_name": str(voice_path),
                "auk_voice_transcript": " \t\n",
            }
        ),
        None,
    )
    assert any(
        issue.short == "voice clone transcript"
        for issue in blank_transcript_issues
    )


@pytest.mark.parametrize("model_class", [AuKBaseServerModel, AuKFlashServerModel])
def test_auk_readiness_checks_every_voice_file_and_server(
    monkeypatch, tmp_path, model_class
):
    expected = ReadinessIssue("server", "unavailable")
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_base_model.SglOmniUtil.check_readiness",
        lambda _: expected,
    )
    first_path = tmp_path / "first.flac"
    first_path.write_bytes(b"audio")
    missing_path = tmp_path / "missing.flac"

    issues = model_class.get_blocking_issues(
        Project.model_validate(
            {
                "auk_voice_file_name": [str(first_path), str(missing_path)],
                "auk_voice_transcript": ["first transcript", "second transcript"],
            }
        ),
        None,
    )

    assert any(issue.short == "voice sample" for issue in issues)
    assert expected in issues


@pytest.mark.parametrize("model_class", [AuKBaseServerModel, AuKFlashServerModel])
def test_auk_generation_builds_voice_clone_payload(
    monkeypatch, tmp_path, model_class
):
    calls = []
    encoded_paths = []
    expected = Sound(np.asarray([], dtype=np.float32), 24_000)

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        calls.append((base_url, payloads, print_request))
        return [expected, expected]

    def fake_make_audio_data_uri(path):
        encoded_paths.append(path)
        return f"data:audio/flac;base64,{path}"

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SglOmniUtil.get_base_url",
        lambda: "http://example.test",
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SoundUtil.make_audio_data_uri",
        fake_make_audio_data_uri,
    )

    first_path = tmp_path / "first.flac"
    second_path_obj = tmp_path / "second.flac"
    project = Project.model_validate(
        {
            "auk_voice_file_name": [str(first_path), str(second_path_obj)],
            "auk_voice_transcript": ["first transcript", "second transcript"],
            "auk_seed": 1234,
        }
    )
    result = model_class().generate_using_project(
        project,
        ["hello", "world"],
        voice_selection_index=1,
        print_generation_request=True,
    )

    second_path = str(tmp_path / "second.flac")
    expected_reference = {
        "audio_path": f"data:audio/flac;base64,{second_path}",
        "text": "second transcript",
    }
    assert result == [expected, expected]
    assert encoded_paths == [second_path]
    assert calls == [
        (
            "http://example.test",
            [
                {
                    "input": "hello",
                    "stream": False,
                    "response_format": "wav",
                    "seed": 1234,
                    "references": [expected_reference],
                },
                {
                    "input": "world",
                    "stream": False,
                    "response_format": "wav",
                    "seed": 1234,
                    "references": [expected_reference],
                },
            ],
            True,
        )
    ]
    assert all("gen_seconds" not in payload for payload in calls[0][1])
    assert all("stage_params" not in payload for payload in calls[0][1])
    assert all("instructions" not in payload for payload in calls[0][1])


def test_auk_speech_speed_inversely_scales_upstream_duration(monkeypatch, tmp_path):
    captured = {}
    reference_sound = Sound(np.zeros(240_000, dtype=np.float32), 24_000)

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SoundFileUtil.load",
        lambda _: reference_sound,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/flac;base64,{path}",
    )

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        captured["payloads"] = payloads
        return []

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )

    project = Project.model_validate(
        {
            "auk_voice_file_name": str(tmp_path / "voice.flac"),
            "auk_voice_transcript": "你好",
            "auk_speed": 2.0,
        }
    )
    AuKBaseServerModel().generate_using_project(project, ["hello!", "hello!!!"])

    assert [
        payload["stage_params"]["auk_engine"]["gen_seconds"]
        for payload in captured["payloads"]
    ] == pytest.approx([5.0, 20.0 / 3.0])


def test_auk_force_random_seed_resolves_once_for_all_prompts(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.random.randrange",
        lambda *_: 9876,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SoundUtil.make_audio_data_uri",
        lambda path: f"data:audio/flac;base64,{path}",
    )

    def fake_generate_concurrent(base_url, payloads, print_request=False):
        captured["payloads"] = payloads
        return []

    monkeypatch.setattr(
        "tts_audiobook_tool.tts_models.auk_server_model.SglOmniUtil.generate_concurrent",
        fake_generate_concurrent,
    )

    project = Project.model_validate(
        {
            "auk_voice_file_name": str(tmp_path / "voice.flac"),
            "auk_voice_transcript": "reference transcript",
            "auk_seed": 1234,
        }
    )
    AuKBaseServerModel().generate_using_project(
        project, ["hello", "world"], force_random_seed=True
    )

    assert [payload["seed"] for payload in captured["payloads"]] == [9876, 9876]


@pytest.mark.parametrize(
    "model_type", [TtsModelType.AUK_SERVER, TtsModelType.AUK_FLASH_SERVER]
)
def test_auk_menu_uses_variant_identity_and_shared_seed(
    monkeypatch, model_type
):
    seen_model_types = []
    seen_number_items = []
    seen_seed_attrs = []
    state = cast(State, SimpleNamespace(project=Project()))

    monkeypatch.setattr(
        VoiceMenuShared,
        "make_voice_sample_items",
        lambda actual_state, actual_type: seen_model_types.append(actual_type) or [],
    )
    monkeypatch.setattr(
        MenuUtil,
        "make_number_item",
        lambda **kwargs: (
            seen_number_items.append(kwargs) or SimpleNamespace(superlabel="")
        ),
    )
    monkeypatch.setattr(
        VoiceMenuShared,
        "make_seed_item",
        lambda actual_state, attr, add_batch_warning=False: (
            seen_seed_attrs.append((attr, add_batch_warning))
            or SimpleNamespace(superlabel="")
        ),
    )
    monkeypatch.setattr(
        VoiceMenuShared,
        "menu_wrapper",
        lambda actual_state, make_items: make_items(actual_state),
    )

    VoiceAuKServerMenu.menu(state, model_type)

    assert seen_model_types == [model_type]
    assert len(seen_number_items) == 1
    assert seen_number_items[0]["attr"] == "auk_speed"
    assert seen_number_items[0]["default_value"] == 1.0
    assert seen_number_items[0]["min_value"] == 0.5
    assert seen_number_items[0]["max_value"] == 2.0
    assert seen_seed_attrs == [("auk_seed", True)]


def test_auk_menu_rejects_non_auk_type():
    with pytest.raises(ValueError, match="Unsupported AuK server type"):
        state = cast(State, SimpleNamespace(project=Project()))
        VoiceAuKServerMenu.menu(state, TtsModelType.QWEN3TTS_SERVER)
