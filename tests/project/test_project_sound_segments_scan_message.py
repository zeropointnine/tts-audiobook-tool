from types import SimpleNamespace
from typing import cast

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_sound_segments import ProjectSoundSegments
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.state import State


def make_sound_segments() -> ProjectSoundSegments:
    sound_segments = object.__new__(ProjectSoundSegments)
    sound_segments.project = cast(Project, object())
    sound_segments._sound_segments_map = {}
    sound_segments._dirty = True
    sound_segments.dont_show_scan_message = False
    return sound_segments


def test_initial_scan_can_suppress_transient_progress_message(monkeypatch) -> None:
    sound_segments = make_sound_segments()
    sound_segments.dont_show_scan_message = True
    expected = {1: []}
    output: list[tuple[object, str]] = []
    monkeypatch.setattr(
        SoundSegmentUtil,
        "make_sound_segments_map",
        lambda project: expected,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.project_support.project_sound_segments.printt",
        lambda value, end="\n": output.append((value, end)),
    )

    result = sound_segments.sound_segments_map

    assert result == expected
    assert output == []
    assert sound_segments._dirty is False


def test_state_enables_scan_messages_after_first_main_menu_display() -> None:
    sound_segments = SimpleNamespace(dont_show_scan_message=True)
    state = object.__new__(State)
    state._project = cast(Project, SimpleNamespace(sound_segments=sound_segments))
    state.dont_show_scan_message = True
    state.has_shown_main_menu = False

    state.mark_main_menu_shown()

    assert state.has_shown_main_menu is True
    assert state.dont_show_scan_message is False
    assert sound_segments.dont_show_scan_message is False
