from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.qwen3_server_model import Qwen3ServerModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.mark.parametrize(
    ("apply_word_substitutions", "expected_prompt"),
    [(True, "bar"), (False, "foo")],
)
def test_server_prompt_is_prepared_exactly_once_and_can_bypass_substitutions(
    monkeypatch,
    apply_word_substitutions: bool,
    expected_prompt: str,
) -> None:
    """A second preparation would turn bar into baz and fail this assertion."""
    project = cast(
        Project,
        SimpleNamespace(
            word_substitutions={"foo": "bar", "bar": "baz"},
            language_code="en",
            qwen3_temperature=-1,
            qwen3_top_p=-1,
            qwen3_top_k=-1,
            qwen3_repetition_penalty=-1,
            qwen3_seed=1,
            on_stream_end=None,
        ),
    )
    model = Qwen3ServerModel()
    captured_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(Tts, "_type", TtsModelType.QWEN3TTS_SERVER)
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: model))
    monkeypatch.setattr(
        ProjectVoiceUtil,
        "current_voice_reference_pair",
        staticmethod(lambda *_args, **_kwargs: ("", "")),
    )
    monkeypatch.setattr(
        SglOmniUtil,
        "generate_concurrent",
        staticmethod(
            lambda _url, payloads, print_request=False: (
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
