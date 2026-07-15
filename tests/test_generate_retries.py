from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil


class StubValidationResult:
    def __init__(self, is_fail: bool) -> None:
        self.is_fail = is_fail
        self.voice_tag = ""

    def get_ui_message_with_extras(self) -> str:
        return "Failed" if self.is_fail else "Passed"


def test_generate_files_retries_validation_failures_up_to_project_limit() -> None:
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        max_retries=2,
        phrase_groups=[phrase_group],
        sound_segments=sound_segments,
    )
    state = SimpleNamespace(
        project=project,
        prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
    )
    results = [StubValidationResult(True), StubValidationResult(True), StubValidationResult(False)]
    force_random_seed_values: list[bool] = []

    def generate_and_validate_batch(**kwargs: object) -> list[StubValidationResult]:
        force_random_seed_values.append(kwargs["force_random_seed"])  # type: ignore[arg-type]
        return [results.pop(0)]

    with patch("tts_audiobook_tool.generate_util.ModelManager.warm_up_models", return_value=SimpleNamespace(should_stop=False)), \
            patch("tts_audiobook_tool.generate_util.readiness.get_generate_blocker_text", return_value=""), \
            patch("tts_audiobook_tool.generate_util.Tts.get_instance", return_value=SimpleNamespace(get_warning_issues=lambda _: [])), \
            patch("tts_audiobook_tool.generate_util.Tts.clear_continuation"), \
            patch("tts_audiobook_tool.generate_util.Tts.reset_voice_rotation_index"), \
            patch("tts_audiobook_tool.generate_util.ProjectVoiceUtil.is_language_cjk", return_value=False), \
            patch("tts_audiobook_tool.generate_util.app_memory.show_vram_memory_warning_if_necessary", return_value=False), \
            patch("tts_audiobook_tool.generate_util.GenerateUtil.generate_and_validate_batch", side_effect=generate_and_validate_batch), \
            patch("tts_audiobook_tool.generate_util.GenerateUtil.save_sound_and_timing_json", return_value=("", "saved.wav")), \
            patch("tts_audiobook_tool.generate_util.Stt.has_instance", return_value=False):
        did_interrupt = GenerateUtil.generate_files(state, {0}, batch_size=1, is_regen=False)

    assert not did_interrupt
    assert force_random_seed_values == [False, True, True]
    assert results == []
