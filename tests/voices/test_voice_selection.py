from unittest.mock import patch
import pytest

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil, TtsModelError
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from project_settings_test_support import set_setting


def make_phrase_group(voice_index: int = 0) -> PhraseGroup:
    return PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=voice_index)


@pytest.fixture(autouse=True)
def initialized_tts_type():
    had_type = hasattr(Tts, "_type")
    previous_type = getattr(Tts, "_type", TtsModelType.require_by_id("mira_local"))
    Tts._type = TtsModelType.require_by_id("mira_local")
    try:
        yield
    finally:
        if not had_type:
            del Tts._type
        else:
            Tts._type = previous_type


@pytest.mark.parametrize("raw_value", ["auto_advance", "disabled", "custom"])
def test_removed_voice_select_mode_is_ignored_on_load_and_not_saved(raw_value: str) -> None:
    # Projects saved before the setting was removed still load; the key is dropped.
    project = Project.model_validate({"voice_select_mode": raw_value})

    payload = ProjectSerializationUtil.to_project_json_dict(project)

    assert "voice_select_mode" not in payload
    assert not hasattr(project, "voice_select_mode")


def run_generation(
        phrase_groups: list[PhraseGroup],
        indices: list[int],
        voices: list[str],
        voice_selection_index: int | None = None,
) -> tuple[int, str]:
    project = Project(tts_model_type="mira_local")
    set_setting(project, "mira_voice_file_name", voices)
    captured_index = -1

    def generate_using_project(*args, **kwargs):
        nonlocal captured_index
        captured_index = kwargs["voice_selection_index"]
        return "test stop"

    with patch("tts_audiobook_tool.generate_util.Tts.get_active_type", return_value=TtsModelType.require_by_id("mira_local")), \
            patch("tts_audiobook_tool.generate_util.Tts.get_instance_if_exists", return_value=None), \
            patch("tts_audiobook_tool.generate_util.Tts.get_voice_tag_for_selection_index", side_effect=lambda _, index: f"voice-{index}") as get_tag, \
            patch("tts_audiobook_tool.generate_util.Tts.generate_using_project", side_effect=generate_using_project):
        results = GenerateUtil.generate(
            project=project,
            indices=indices,
            phrase_groups=phrase_groups,
            force_random_seed=False,
            is_realtime=False,
            save_debug_files=False,
            voice_selection_index=voice_selection_index,
        )

    assert all(isinstance(result, TtsModelError) for result in results)
    tagged_index = get_tag.call_args.args[1]
    return captured_index, f"voice-{tagged_index}"


@pytest.mark.parametrize(
    ("requested_index", "expected_index"),
    [(-2, 0), (-1, 0), (2, 0), (100, 0)],
)
def test_line_voice_invalid_index_defaults_to_first_voice(
        requested_index: int,
        expected_index: int,
) -> None:
    index, voice_tag = run_generation(
        [make_phrase_group(requested_index)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag) == (expected_index, f"voice-{expected_index}")


def test_line_voice_uses_explicit_phrase_voice_index() -> None:
    index, voice_tag = run_generation(
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag) == (1, "voice-1")


def test_line_voice_without_voice_samples_uses_the_default_voice_slot() -> None:
    # A project always has at least one voice; with no samples configured that
    # is one sample-less voice, so the (stale) phrase index defaults to it.
    index, voice_tag = run_generation(
        [make_phrase_group(1)],
        [0],
        [],
    )

    assert (index, voice_tag) == (0, "voice-0")


def test_line_voice_multi_item_batch_with_explicit_voice_uses_it() -> None:
    index, voice_tag = run_generation(
        [make_phrase_group(0), make_phrase_group(1)],
        [0, 1],
        ["voice-a.flac", "voice-b.flac"],
        voice_selection_index=1,
    )

    assert (index, voice_tag) == (1, "voice-1")


def test_line_voice_multi_item_batch_without_explicit_voice_raises() -> None:
    with pytest.raises(ValueError, match="voice_selection_index"):
        run_generation(
                [make_phrase_group(0), make_phrase_group(1)],
            [0, 1],
            ["voice-a.flac", "voice-b.flac"],
        )


def test_line_voice_selection_is_stable_across_realtime_style_retries() -> None:
    first = run_generation(
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )
    retry = run_generation(
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert first == retry == (1, "voice-1")
