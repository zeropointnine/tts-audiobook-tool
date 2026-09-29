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
    project = Project()
    project.dir_path = str(tmp_path)
    project.word_substitutions = {"foo": "bar", "bar": "baz"}
    project.language_code = "en"
    project.on_stream_end = None
    project.set_model_setting("server_qwen3tts", "file_name", ["sample.flac"])
    definition = load_definitions().models["server_qwen3tts"]
    model = SglOmniBackendAdapter(definition, ConfiguredModelSupport(definition))
    captured_payloads: list[dict[str, object]] = []

    from tts_audiobook_tool.sound.sound_util import SoundUtil
    monkeypatch.setattr(SoundUtil, "make_audio_data_uri", staticmethod(lambda path: "data:" + path))

    monkeypatch.setattr(Tts, "_type", TtsModelType.QWEN3TTS_SERVER)
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
