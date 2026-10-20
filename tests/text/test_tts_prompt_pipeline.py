from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.sgl_omni_configured import (
    ConfiguredModelSupport,
    SglOmniBackendAdapter,
)
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.mark.parametrize(
    ("apply_word_substitutions", "expected_prompt"),
    [(True, "bar"), (False, "foo")],
)
def test_server_prompt_is_prepared_exactly_once_and_can_bypass_substitutions(
    monkeypatch,
    tmp_path,
    apply_word_substitutions: bool,
    expected_prompt: str,
) -> None:
    """A second preparation would turn bar into baz and fail this assertion."""
    sample = tmp_path / "sample.flac"
    sample.write_bytes(b"audio")
    # No watcher is needed for this temporary reference; avoid consuming
    # inotify instances in long test runs.
    project = Project(tts_model_type="qwen3tts_sglomni")
    project.dir_path = str(tmp_path)
    project.word_substitutions = {"foo": "bar", "bar": "baz"}
    project.language_code = "en"
    project.on_stream_end = None
    project.set_model_setting("qwen3tts_sglomni", "file_name", ["sample.flac"])
    definition = load_definitions().models["qwen3tts_sglomni"]
    model = SglOmniBackendAdapter(definition, ConfiguredModelSupport(definition))
    captured_payloads: list[dict[str, object]] = []

    from tts_audiobook_tool.sound.sound_util import SoundUtil
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", staticmethod(lambda path: "data:" + path))

    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("qwen3tts_sglomni"))
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: model))
    monkeypatch.setattr(
        ProjectVoiceUtil,
        "current_voice_reference_pair",
        staticmethod(lambda *_args, **_kwargs: ("sample.flac", "")),
    )
    monkeypatch.setattr(
        SglOmniUtil,
        "generate_concurrent",
        staticmethod(
            lambda _url, payloads, **_kwargs: (
                captured_payloads.extend(payloads) or []
            )
        ),
    )
    monkeypatch.setattr(
        SglOmniUtil, "get_base_url", staticmethod(lambda: "http://example.test")
    )

    result = Tts.generate_using_project(
        project,
        ["foo"],
        voice_selection_index=0,
        apply_word_substitutions=apply_word_substitutions,
    )

    assert result == []
    assert captured_payloads[0]["input"] == expected_prompt


@pytest.mark.parametrize("model_id, caller_cap, expected", [
    ("chatterbox_local", -1, -1),
    ("chatterbox_local", 10, 10),
    ("qwen3tts_sglomni", -1, -1),
    ("qwen3tts_sglomni", 10, 10),
    ("echo_tts_audiocpp", -1, 2147483647),
    ("echo_tts_audiocpp", 10, 10),
    ("echo_tts_audiocpp", 0, 0),
    ("echo_tts_audiocpp", 2**32 - 1, 2147483647),
    ("moss_delay_audiocpp", -1, 2147483647),
    ("moss_delay_audiocpp", 10, 10),
    ("moss_delay_audiocpp", 0, 0),
    ("moss_delay_audiocpp", 2**32 - 1, 2147483647),
    ("moss_local_audiocpp", -1, -1),
])
def test_dispatch_forwards_catalog_and_caller_random_seed_caps(monkeypatch, model_id, caller_cap, expected):
    model_type = TtsModelType.require_by_id(model_id)
    project = Project(tts_model_type=model_id)
    captured = []

    def generate(**kwargs):
        captured.append(kwargs)
        return []

    model = SimpleNamespace(
        prepare_text_for_inference=lambda _project, text, **_kwargs: text,
        generate_using_project=generate,
    )
    monkeypatch.setattr(Tts, "_type", model_type)
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: model))

    assert Tts.generate_using_project(project, ["hello"], max_random_seed=caller_cap) == []
    assert captured[0]["max_random_seed"] == expected
    assert captured[0]["prompts"] == ["hello"]


def test_dispatch_uses_live_catalog_cap_not_cached_instance_metadata(monkeypatch):
    model_type = TtsModelType.require_by_id("chatterbox_local")
    cached_spec = model_type.value
    captured = []
    model = SimpleNamespace(
        INFO=cached_spec,
        prepare_text_for_inference=lambda _project, text, **_kwargs: text,
        generate_using_project=lambda **kwargs: captured.append(kwargs) or [],
    )
    monkeypatch.setitem(TtsModelType._specs, model_type.id, cached_spec._replace(max_random_seed=7))
    monkeypatch.setattr(Tts, "_type", model_type)
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: model))

    assert Tts.generate_using_project(Project(tts_model_type=model_type.id), ["hello"]) == []
    assert captured[0]["max_random_seed"] == 7
    assert model.INFO.max_random_seed == -1


@pytest.mark.parametrize("invalid", [True, -2, 1.5, "10"])
def test_dispatch_rejects_invalid_caps_before_generation(monkeypatch, invalid):
    model_type = TtsModelType.require_by_id("echo_tts_audiocpp")
    monkeypatch.setattr(Tts, "_type", model_type)
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: pytest.fail("must not generate")))

    result = Tts.generate_using_project(Project(tts_model_type=model_type.id), ["hello"], max_random_seed=invalid)
    assert "max_random_seed" in result
