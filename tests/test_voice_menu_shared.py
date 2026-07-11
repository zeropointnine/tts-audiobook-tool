from unittest.mock import patch

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.menus.voice import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def make_project(tmp_path) -> Project:
    project = Project(dir_path=str(tmp_path))
    project._autosave = False
    return project


def test_manage_voice_samples_label_zero_one_and_multiple(tmp_path):
    project = make_project(tmp_path)
    state = type("StateStub", (), {"project": project})()
    item = VoiceMenuShared.make_manage_voice_samples_item(
        state,
        TtsModelType.OMNIVOICE,
        no_samples_label=lambda _: "Select voice clone sample (currently: none)",
    )

    assert item.label(state) == "Select voice clone sample (currently: none)"

    project.omnivoice_voice_file_name = ["zzz belle 24b 20s_omnivoice.flac"]
    assert "Manage voice clone sample/s" in item.label(state)
    assert "zzz belle 24b 20s" in item.label(state)
    assert "+" not in item.label(state)

    project.omnivoice_voice_file_name = [
        "zzz belle 24b 20s_omnivoice.flac",
        "zzz belle 24c 20s_omnivoice.flac",
        "a_third_voice_file_omnivoice.flac",
    ]
    assert "zzz belle 24b 20s, +2 more" in item.label(state)


def test_make_voice_samples_subheading_always_shows_rotation_copy(tmp_path):
    project = make_project(tmp_path)

    project.omnivoice_voice_file_name = ["one_omnivoice.flac"]
    assert VoiceMenuShared.make_voice_samples_subheading(project, TtsModelType.OMNIVOICE) == (
        "Multiple samples cycle in order, one per generation.\n"
        "\n"
        "- 1) one\n"
    )

    project.omnivoice_voice_file_name = [
        "one_omnivoice.flac",
        "two_omnivoice.flac",
    ]
    assert VoiceMenuShared.make_voice_samples_subheading(project, TtsModelType.OMNIVOICE) == (
        "Multiple samples cycle in order, one per generation.\n"
        "\n"
        "- 1) one\n"
        "- 2) two\n"
    )


def test_set_voice_and_save_append_keeps_existing_parallel_lists(tmp_path):
    project = make_project(tmp_path)
    project.omnivoice_voice_file_name = ["existing_omnivoice.flac"]
    project.omnivoice_voice_transcript = ["existing transcript"]
    sound = Sound(data=[], sr=24000)

    with patch("tts_audiobook_tool.project_support.project_voice_util.SoundFileUtil.save_flac", return_value=""):
        err = ProjectVoiceUtil.set_voice_and_save(
            project,
            sound,
            "added",
            "added transcript",
            TtsModelType.OMNIVOICE,
            append=True,
        )

    assert err == ""
    assert project.omnivoice_voice_file_name == ["existing_omnivoice.flac", "added_omnivoice.flac"]
    assert project.omnivoice_voice_transcript == ["existing transcript", "added transcript"]


def test_remove_single_voice_sample_returns_one_level(tmp_path):
    project = make_project(tmp_path)
    project.omnivoice_voice_file_name = ["only_omnivoice.flac"]
    project.omnivoice_voice_transcript = ["only transcript"]
    state = type("StateStub", (), {"project": project})()

    with patch("tts_audiobook_tool.menus.voice.voice_menu_shared.print_feedback") as print_feedback:
        should_return = VoiceMenuShared.remove_voice_sample_from_menu(state, TtsModelType.OMNIVOICE)

    assert should_return is True
    assert project.omnivoice_voice_file_name == []
    assert project.omnivoice_voice_transcript == []
    print_feedback.assert_called_with("Removed only")


def test_remove_multi_voice_sample_bad_value_stays(tmp_path):
    project = make_project(tmp_path)
    project.omnivoice_voice_file_name = ["one_omnivoice.flac", "two_omnivoice.flac"]
    state = type("StateStub", (), {"project": project})()

    with patch("tts_audiobook_tool.menus.voice.voice_menu_shared.ask.ask", return_value="bad"), \
            patch("tts_audiobook_tool.menus.voice.voice_menu_shared.print_feedback") as print_feedback:
        should_return = VoiceMenuShared.remove_voice_sample_from_menu(state, TtsModelType.OMNIVOICE)

    assert should_return is False
    assert project.omnivoice_voice_file_name == ["one_omnivoice.flac", "two_omnivoice.flac"]
    print_feedback.assert_called_with("Bad value", is_error=True)


def test_remove_multi_voice_sample_removes_selected_parallel_transcript_and_stays(tmp_path):
    project = make_project(tmp_path)
    project.omnivoice_voice_file_name = [
        "one_omnivoice.flac",
        "two_omnivoice.flac",
        "three_omnivoice.flac",
    ]
    project.omnivoice_voice_transcript = ["one text", "two text", "three text"]
    state = type("StateStub", (), {"project": project})()

    with patch("tts_audiobook_tool.menus.voice.voice_menu_shared.ask.ask", return_value="2"), \
            patch("tts_audiobook_tool.menus.voice.voice_menu_shared.print_feedback") as print_feedback:
        should_return = VoiceMenuShared.remove_voice_sample_from_menu(state, TtsModelType.OMNIVOICE)

    assert should_return is False
    assert project.omnivoice_voice_file_name == ["one_omnivoice.flac", "three_omnivoice.flac"]
    assert project.omnivoice_voice_transcript == ["one text", "three text"]
    print_feedback.assert_called_with("Removed two")
