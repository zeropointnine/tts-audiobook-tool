from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types import (
    ExportType,
    HighShelfEq,
    NormalizationType,
    RoomTone,
    SectionMarkerMode,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus import concat_menu
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil
from tts_audiobook_tool.state import State


def make_phrase_group(text: str) -> PhraseGroup:
    return PhraseGroup([Phrase(text, Reason.SENTENCE)])


# ---------------------------------------------------------------------------
# Output indices dialog
# ---------------------------------------------------------------------------


def test_ask_output_indices_empty_input_cancels() -> None:
    infos = [
        SimpleNamespace(output_index=0, num_files_exist=1),
        SimpleNamespace(output_index=1, num_files_exist=0),
    ]

    with patch.object(concat_menu.ask, "ask_input", return_value=""), patch.object(
        concat_menu, "printt"
    ), patch.object(concat_menu, "print_feedback") as print_feedback:
        result = concat_menu.ask_output_indices(infos)  # type: ignore[arg-type]

    assert result is None
    print_feedback.assert_not_called()


def test_enabled_option_confirmation_lines_only_include_enabled_options() -> None:
    project = Project.model_validate({})
    project.normalization_type = NormalizationType.STRONGER
    project.use_upsampler = True
    project.use_break_sound_effect = True
    project.high_shelf = HighShelfEq.MODERATE.id
    state = cast(State, SimpleNamespace(project=project))

    assert concat_menu.make_enabled_option_confirmation_lines(state) == [
        "- Loudness normalization: Stronger",
        "- Generative upsampling: True",
        "- Section break sound effects: True",
        "- Treble uplift: Moderate",
    ]

    project.normalization_type = NormalizationType.DEFAULT
    assert concat_menu.make_enabled_option_confirmation_lines(state)[0] == (
        "- Loudness normalization: ACX standard"
    )

    project.high_shelf = HighShelfEq.STRONGER.id
    assert concat_menu.make_enabled_option_confirmation_lines(state)[-1] == (
        "- Treble uplift: Stronger"
    )

    project.limit_silence_gaps = True
    project.limit_silence_gaps_duration = 0.35
    assert concat_menu.make_enabled_option_confirmation_lines(state)[-1] == (
        "- Maximum silence inside segments: True 0.35s"
    )

    project.normalization_type = NormalizationType.DISABLED
    project.use_upsampler = False
    project.use_break_sound_effect = False
    project.high_shelf = HighShelfEq.DISABLED.id
    project.limit_silence_gaps = False

    assert concat_menu.make_enabled_option_confirmation_lines(state) == []


def test_ask_output_indices_and_make_single_file_uses_markers_as_bookmarks_for_single_book_section() -> None:
    project = Project.model_validate({
        "phrase_groups": [
            make_phrase_group("One."),
            make_phrase_group("Two."),
            make_phrase_group("Three."),
        ],
        "markers": [1, 2],
        "book": Book(sections=[BookSection(title="Chapter 1", phrase_groups=[
            make_phrase_group("One."),
            make_phrase_group("Two."),
            make_phrase_group("Three."),
        ])]),
    })
    state = SimpleNamespace(
        project=project,
        prefs=SimpleNamespace(project_dir="/tmp"),
    )
    state.project.markers = [1, 2]
    state.project.export_type = ExportType.AAC
    state.project.chapter_mode = SectionMarkerMode.BOOKMARKS
    state.project._sound_segments = SimpleNamespace(num_generated=lambda: 1)
    events: list[str] = []

    with patch.object(
        concat_menu.ask,
        "ask_confirm",
        side_effect=lambda: events.append("confirm") or True,
    ), patch.object(
        concat_menu.OutputRangeInfo,
        "make_single_info",
        return_value=SimpleNamespace(num_files_exist=1, num_segments=3),
    ), patch.object(
        concat_menu.MenuUtil,
        "print_screen_heading",
        side_effect=lambda *_: events.append("heading"),
    ) as print_screen_heading, patch.object(
        concat_menu.ConcatUtil,
        "make_files",
    ) as make_files_mock, patch.object(
        concat_menu,
        "printt",
        side_effect=lambda value="": events.append(value),
    ):
        concat_menu.ask_output_indices_and_make(cast(State, state))

    print_screen_heading.assert_called_once_with(state, "Start")
    assert events.index("heading") < events.index("Will create a single AAC/M4B file")
    assert events.index("Will create a single AAC/M4B file") < events.index("confirm")
    make_files_mock.assert_called_once_with(
        state=state,
        file_cut_indices=[],
        bookmark_indices=[1, 2],
    )


def test_ask_output_indices_and_make_single_file_ignores_markers_as_bookmarks_for_multiple_book_sections() -> None:
    project = Project.model_validate({
        "phrase_groups": [
            make_phrase_group("One."),
            make_phrase_group("Two."),
            make_phrase_group("Three."),
        ],
        "markers": [1, 2],
        "book": Book(sections=[
            BookSection(title="Chapter 1", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Chapter 2", phrase_groups=[make_phrase_group("Two."), make_phrase_group("Three.")]),
        ]),
    })
    state = SimpleNamespace(
        project=project,
        prefs=SimpleNamespace(project_dir="/tmp"),
    )
    state.project.markers = [1, 2]
    state.project.export_type = ExportType.AAC
    state.project.chapter_mode = SectionMarkerMode.BOOKMARKS
    state.project._sound_segments = SimpleNamespace(num_generated=lambda: 1)

    with patch.object(concat_menu.ask, "ask_confirm", return_value=True), \
        patch.object(concat_menu.OutputRangeInfo, "make_single_info", return_value=SimpleNamespace(num_files_exist=1, num_segments=3)), \
            patch.object(concat_menu.MenuUtil, "print_screen_heading") as print_screen_heading, \
            patch.object(concat_menu.ConcatUtil, "make_files") as make_files_mock, \
            patch.object(concat_menu, "printt"):
        concat_menu.ask_output_indices_and_make(cast(State, state))

    print_screen_heading.assert_called_once_with(state, "Start")
    make_files_mock.assert_called_once_with(
        state=state,
        file_cut_indices=[],
        bookmark_indices=[],
    )


# ---------------------------------------------------------------------------
# Generative upsampling (LavaSR) menu items
# ---------------------------------------------------------------------------


def test_concat_menu_always_shows_generative_upsampling() -> None:
    project = Project.model_validate({})
    prefs = SimpleNamespace(aac_bitrate="128k")
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))

    with patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.menu"
    ) as menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
        return_value=[],
    ), patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
        ConcatMenu.menu(state)
        assert menu.call_args.args[1] == "Create audiobook file/s"
        assert menu.call_args.kwargs.get("breadcrumb") is None
        make_items = menu.call_args.args[2]
        items = make_items(state)

    labels = [get_string_from(state, item.label) for item in items]
    assert any(label.startswith("Generative upsampling") for label in labels)


def test_concat_menu_offers_limit_silence_gaps() -> None:
    project = Project.model_validate({"limit_silence_gaps": True, "limit_silence_gaps_duration": 0.4})
    state = cast(State, SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="128k")))

    with patch("tts_audiobook_tool.menus.concat_menu.MenuUtil.menu") as menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
        return_value=[],
    ), patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
        ConcatMenu.menu(state)
        items = menu.call_args.args[2](state)

    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    assert "Maximum silence inside segments (currently: enabled 0.40s)" in labels

    project.limit_silence_gaps = False
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    assert "Maximum silence inside segments (currently: disabled)" in labels


def test_limit_silence_gaps_menu_writes_project_setting() -> None:
    project = Project.model_validate({"limit_silence_gaps": False})
    state = cast(State, SimpleNamespace(project=project))

    # The first item cycles disabled/enabled; the Duration item (shown with a
    # trailing "s" unit) only exists while the setting is enabled
    with patch.object(concat_menu.MenuUtil, "menu") as menu, patch.object(
        Project, "save"
    ) as save, patch.object(concat_menu, "print_feedback"):
        ConcatMenu.limit_silence_gaps_menu(state)
        subheading = menu.call_args.kwargs["subheading"]
        make_items = menu.call_args.kwargs["items"]

        items = make_items(state)
        assert len(items) == 1
        label = strip_ansi_codes(get_string_from(state, items[0].label))
        assert label == "Value: disabled (default)"

        items[0].handler(state, items[0])
        assert project.limit_silence_gaps
        save.assert_called_once()

        items = make_items(state)
        assert len(items) == 2
        assert strip_ansi_codes(get_string_from(state, items[0].label)) == "Value: enabled"
        assert strip_ansi_codes(get_string_from(state, items[1].label)).startswith(
            "Duration (currently: 1.00s default)"
        )

        items[0].handler(state, items[0])
        assert not project.limit_silence_gaps
        assert len(make_items(state)) == 1

    assert "realtime playback" in subheading


@pytest.mark.parametrize("sample_rate", [24000, 48000])
def test_upsampling_subheading_has_no_terminal_newline_and_preserves_warning_gap(monkeypatch, sample_rate):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="chatterbox_local")))
    support = SimpleNamespace(get_output_sample_rate=lambda _: sample_rate)
    captured = {}
    monkeypatch.setattr(ModelWorker, "probe_lava_sr_blocking", lambda: (True, ""))
    monkeypatch.setattr(concat_menu.Tts, "get_model_support", lambda _: support)
    monkeypatch.setattr(concat_menu.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))

    ConcatMenu.upsample_menu(state)

    text = strip_ansi_codes(captured["subheading"])
    assert not text.endswith("\n")
    if sample_rate >= 44100:
        assert "\n\n* " in text
        assert f"samplerate of {sample_rate}." in text
    else:
        assert "already outputs" not in text


def test_concat_menu_prevents_enabling_unavailable_lava_sr() -> None:
    project = Project.model_validate({"use_upsampler": False})
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(ModelWorker, "probe_lava_sr_blocking", return_value=(False, "")), patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.options_menu"
    ) as options_menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ask.ask_error"
    ) as ask_error, patch.object(Project, "save") as save:
        ConcatMenu.upsample_menu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](True)

    assert "LavaSR v2 upsampler not installed" in kwargs["subheading"]
    assert not kwargs["subheading"].endswith("\n")
    assert not project.use_upsampler
    save.assert_not_called()
    ask_error.assert_called_once_with(
        "LavaSR v2 is not installed; generative upsampling cannot be enabled"
    )


def test_min_segment_sentence_silence_menu_cycles_and_saves() -> None:
    # The submenu has one cycling item; selecting it toggles the project value and saves
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch
    from tts_audiobook_tool.app_types import MinSegmentSentenceSilence
    from tts_audiobook_tool.menus.concat_menu import ConcatMenu
    from tts_audiobook_tool.menus.menu_util import get_string_from
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.state import State

    project = Project.model_validate({})
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, \
            patch("tts_audiobook_tool.menus.concat_menu.MenuUtil.menu") as menu, \
            patch("tts_audiobook_tool.menus.concat_menu.print_feedback"):
        ConcatMenu.min_segment_sentence_silence_menu(state)
        kwargs = menu.call_args.kwargs
        assert kwargs["heading"] == "Minimum sentence silence inside segments"
        assert "(0.5s)" in kwargs["subheading"]
        (item,) = kwargs["items"]

        assert "disabled" in get_string_from(state, item.label)
        assert "(default)" in get_string_from(state, item.label)

        item.handler(state, item)
        assert project.min_segment_sentence_silence is MinSegmentSentenceSilence.ENABLED
        assert "(default)" not in get_string_from(state, item.label)
        save.assert_called_once()

        item.handler(state, item)
        assert project.min_segment_sentence_silence is MinSegmentSentenceSilence.DISABLED


def test_concat_menu_offers_room_tone_after_prosody_modulation() -> None:
    project = Project()
    state = cast(State, SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="128k")))

    with patch("tts_audiobook_tool.menus.concat_menu.MenuUtil.menu") as menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
        return_value=[],
    ), patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
        ConcatMenu.menu(state)
        items = menu.call_args.args[2](state)

    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    index = labels.index("Room tone (currently: Off)")
    assert labels[index - 1].startswith("Prosody modulation")


def test_room_tone_menu_writes_project_setting() -> None:
    project = Project()
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(concat_menu.MenuUtil, "options_menu") as options_menu, patch.object(
        Project, "save"
    ) as save, patch.object(concat_menu, "print_feedback"):
        ConcatMenu.room_tone_menu(state)
        kwargs = options_menu.call_args.kwargs
        assert kwargs["heading_text"] == "Room tone"
        assert kwargs["labels"] == ["Off", "Low", "Medium", "High"]
        assert kwargs["values"] == list(RoomTone)
        assert kwargs["sublabels"][1:] == ["-75 dBFS", "-70 dBFS", "-65 dBFS"]
        assert kwargs["current_value"] is RoomTone.OFF
        assert kwargs["default_value"] is RoomTone.OFF
        assert kwargs["subheading"]

        kwargs["on_select"](RoomTone.MEDIUM)

    assert project.room_tone is RoomTone.MEDIUM
    save.assert_called_once()
