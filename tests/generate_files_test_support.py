"""
Shared test support for GenerateUtil.generate_files() tests.

Replaces the model/validation layer with a mocked stack so tests can inject
a deterministic generate_and_validate_batch side effect. Call
generate_files_mock_stack() as a context manager around
GenerateUtil.generate_files(...). make_generate_state() builds the minimal
State those tests drive.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import MagicMock, patch

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def make_generate_state(num_groups: int = 2, max_retries: int = 2) -> State:
    phrase_groups = [
        PhraseGroup([Phrase(f"Hello world {i}.", Reason.SENTENCE)])
        for i in range(num_groups)
    ]
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        max_retries=max_retries,
        phrase_groups=phrase_groups,
        sound_segments=sound_segments,
        generate_range_string="all",
        voice_select_mode=VoiceSelectMode.DISABLED,
        save=MagicMock(return_value=""),
    )
    return cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
        ),
    )


class StubValidationResult:
    def __init__(self, is_fail: bool) -> None:
        self.is_fail = is_fail
        self.voice_tag = ""

    def get_ui_message_with_extras(self) -> str:
        return "Failed" if self.is_fail else "Passed"


@contextmanager
def generate_files_mock_stack(
        batch_side_effect: Any,
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
        # Run warnings (incl. the voice-variation note) read real project voice data
        patch("tts_audiobook_tool.generate_util.get_run_warnings", return_value=[]),
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

    for patcher in patches:
        patcher.start()
    try:
        yield
    finally:
        for patcher in reversed(patches):
            patcher.stop()