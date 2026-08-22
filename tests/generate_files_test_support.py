"""
Shared test support for GenerateUtil.generate_files() tests.

Replaces the model/validation layer with a mocked stack so tests can inject
a deterministic generate_and_validate_batch side effect. Call
generate_files_mock_stack() as a context manager around
GenerateUtil.generate_files(...).
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class StubValidationResult:
    def __init__(self, is_fail: bool) -> None:
        self.is_fail = is_fail
        self.voice_tag = ""

    def get_ui_message_with_extras(self) -> str:
        return "Failed" if self.is_fail else "Passed"


@contextmanager
def generate_files_mock_stack(
        batch_side_effect: Any,
        model_type: TtsModelType | None = None,
) -> Iterator[None]:
    patches = [
        patch(
            "tts_audiobook_tool.generate_util.ModelManager.warm_up_models",
            return_value=SimpleNamespace(should_stop=False),
        ),
        patch(
            "tts_audiobook_tool.generate_util.readiness.get_generate_blocker_text",
            return_value="",
        ),
        patch(
            "tts_audiobook_tool.generate_util.Tts.get_instance",
            return_value=SimpleNamespace(get_warning_issues=lambda _: []),
        ),
        patch("tts_audiobook_tool.generate_util.Tts.clear_continuation"),
        patch("tts_audiobook_tool.generate_util.Tts.reset_voice_selection_index"),
        patch(
            "tts_audiobook_tool.generate_util.ProjectVoiceUtil.is_language_cjk",
            return_value=False,
        ),
        patch(
            "tts_audiobook_tool.generate_util"
            ".app_memory.show_vram_memory_warning_if_necessary",
            return_value=False,
        ),
        patch(
            "tts_audiobook_tool.generate_util.GenerateUtil.generate_and_validate_batch",
            side_effect=batch_side_effect,
        ),
        patch(
            "tts_audiobook_tool.generate_util.GenerateUtil.save_sound_and_timing_json",
            return_value=("", "saved.wav"),
        ),
        patch("tts_audiobook_tool.generate_util.Stt.has_instance", return_value=False),
    ]
    if model_type is not None:
        patches.insert(
            3,
            patch(
                "tts_audiobook_tool.generate_util.Tts.get_type",
                return_value=model_type,
            ),
        )

    for patcher in patches:
        patcher.start()
    try:
        yield
    finally:
        for patcher in reversed(patches):
            patcher.stop()