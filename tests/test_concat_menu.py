from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import Book, BookSection, VoiceSelectMode
from tts_audiobook_tool.app_types import (
    ExportType,
    HighShelfEq,
    MinSegmentSentenceSilence,
    NormalizationType,
    RoomReverb,
    RoomReverbStereo,
    RoomTone,
    SectionMarkerMode,
    VoiceSelectMode,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice
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


def test_start_shows_split_file_list_before_file_number_prompt() -> None:
    # Start renders the same range/progress description as Create, before input.
    state = cast(State, SimpleNamespace(
        project=SimpleNamespace(
            export_type=ExportType.AAC,
            chapter_mode=SectionMarkerMode.FILES,
            markers=[9],
            sound_segments=SimpleNamespace(num_generated=lambda: 1021),
        ),
        prefs=SimpleNamespace(project_dir="/tmp"),
    ))
    infos = [
        concat_menu.OutputRangeInfo(0, 0, 8, {}),
        concat_menu.OutputRangeInfo(1, 9, 6946, dict.fromkeys(range(9, 1030), "audio.wav")),
    ]
    events: list[str] = []

    def record_print(value: str = "") -> None:
        events.append(strip_ansi_codes(value))

    with patch.object(
        concat_menu.OutputRangeInfo, "make_output_range_infos", return_value=infos
    ), patch.object(
        concat_menu.MenuUtil, "print_heading",
        side_effect=lambda *_args, **_kwargs: events.append("Start"),
    ), patch(
        "tts_audiobook_tool.menus.menu_util.printt", side_effect=record_print
    ), patch.object(
        concat_menu, "printt", side_effect=record_print
    ), patch.object(
        concat_menu.ask, "ask_input", return_value=""
    ), patch.object(concat_menu.ConcatUtil, "make_files") as make_files:
        expected_description = strip_ansi_codes(concat_menu.make_output_files_subheading(state))
        concat_menu.ask_output_indices_and_make(state)

    assert expected_description == (
        "▸ File 1: lines 1 to 9 (0/9 generated)\n"
        "▸ File 2: lines 10 to 6947 (1021/6938 generated)"
    )
    assert events == [
        "Start",
        expected_description,
        "",
        "Enter file numbers to create:",
        '(For example: "1, 2, 4" or  "2-5", or "all")',
    ]
    make_files.assert_not_called()


def plain_option_lines(state: State) -> list[str]:
    return [strip_ansi_codes(line) for line in concat_menu.make_enabled_option_confirmation_lines(state)]


def test_enabled_option_confirmation_lines_only_include_enabled_options() -> None:
    project = Project.model_validate({})
    project.normalization_type = NormalizationType.STRONGER
    project.use_upsampler = True
    project.use_break_sound_effect = True
    project.high_shelf = HighShelfEq.MODERATE.id
    state = cast(State, SimpleNamespace(project=project))

    assert plain_option_lines(state) == [
        "  - Section break sound effects: True",
        "  - Loudness normalization: Stronger",
        "  - Generative upsampling: True",
        "  - Treble EQ: Moderate",
        "  - Channel output: Mono",
    ]

    project.room_tone = RoomTone.MEDIUM
    # Room reverb is per voice; voice 0's is the one applied
    project.voice_references = [Voice("v.flac", room_reverb=RoomReverb.MEDIUM_DRR21)]
    project.room_reverb_stereo = RoomReverbStereo.LATE_W100
    project.limit_silence_gaps = True
    project.limit_silence_gaps_duration = 1.0
    project.min_segment_sentence_silence = MinSegmentSentenceSilence.ENABLED
    project.high_shelf = HighShelfEq.DISABLED.id
    project.normalization_type = NormalizationType.DEFAULT
    # Same order as the menu: options, then post-processing
    assert plain_option_lines(state) == [
        "  - Maximum silence inside segments: True 1.00s",
        "  - Minimum sentence silence inside segments: True",
        "  - Section break sound effects: True",
        "  - Loudness normalization: ACX standard",
        "  - Generative upsampling: True",
        "  - Room reverb: Larger room",
        "  - Room reverb stereo effect: Wide",
        "  - Room tone: Medium",
        "  - Channel output: Stereo",
    ]
    project.room_tone = RoomTone.OFF
    project.voice_references = []
    project.min_segment_sentence_silence = MinSegmentSentenceSilence.DISABLED
    project.limit_silence_gaps = False
    project.high_shelf = HighShelfEq.MODERATE.id

    assert plain_option_lines(state)[1] == (
        "  - Loudness normalization: ACX standard"
    )

    project.high_shelf = HighShelfEq.STRONGER.id
    assert plain_option_lines(state)[-2:] == [
        "  - Treble EQ: Stronger", "  - Channel output: Mono",
    ]

    project.limit_silence_gaps = True
    project.limit_silence_gaps_duration = 0.35
    assert plain_option_lines(state)[0] == (
        "  - Maximum silence inside segments: True 0.35s"
    )

    project.normalization_type = NormalizationType.DISABLED
    project.use_upsampler = False
    project.use_break_sound_effect = False
    project.high_shelf = HighShelfEq.DISABLED.id
    project.limit_silence_gaps = False

    assert plain_option_lines(state) == ["  - Channel output: Mono"]

    # The confirmation reflects the shared channel policy, not voice counts.
    with patch.object(concat_menu.VoiceSpatialPolicy, "from_project", return_value=SimpleNamespace(
        requested_channels=2, room_reverb_summary=RoomReverb.OFF, stereo_position_summary=0.0,
    )):
        assert plain_option_lines(state) == ["  - Channel output: Stereo"]

    # Stereo position, listed before room reverb (same order as the voice sample submenu)
    project.voice_references = [Voice("a.flac", pan=-0.3, room_reverb=RoomReverb.SMALL_DRR22)]
    assert plain_option_lines(state) == [
        "  - Stereo position: Left 30",
        "  - Room reverb: Small room, light",
        "  - Room reverb stereo effect: Wide",
        "  - Channel output: Stereo",
    ]
    project.voice_references = [Voice("a.flac", pan=-0.3), Voice("b.flac")]
    project.voice_select_mode = VoiceSelectMode.USER_DEFINED
    assert plain_option_lines(state) == [
        "  - Stereo position: Varies by voice", "  - Channel output: Stereo",
    ]

    # Each voice's own room is applied; differing voices are summarized
    project.voice_select_mode = VoiceSelectMode.USER_DEFINED
    project.voice_references = [Voice("a.flac", room_reverb=RoomReverb.SMALL_DRR22), Voice("b.flac")]
    assert plain_option_lines(state) == [
        "  - Room reverb: Varies by voice", "  - Room reverb stereo effect: Wide", "  - Channel output: Stereo",
    ]
    # ... but only voice 0 is heard outside user-defined voice selection
    project.voice_select_mode = VoiceSelectMode.AUTO_ADVANCE
    assert plain_option_lines(state)[0] == "  - Room reverb: Small room, light"
    project.voice_references = [Voice("a.flac"), Voice("b.flac", room_reverb=RoomReverb.SMALL_DRR22)]
    assert plain_option_lines(state) == ["  - Channel output: Mono"]


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
        side_effect=lambda *_args, **_kwargs: events.append("heading"),
    ) as print_screen_heading, patch.object(
        concat_menu.ConcatUtil,
        "make_files",
    ) as make_files_mock, patch.object(
        concat_menu,
        "printt",
        side_effect=lambda value="": events.append(value),
    ), patch.object(
        concat_menu,
        "print_accent_separator",
        side_effect=lambda message, **_kwargs: events.append(message),
    ):
        concat_menu.ask_output_indices_and_make(cast(State, state))

    print_screen_heading.assert_called_once_with(
        state, "Start", subheading=concat_menu.make_output_files_subheading
    )
    assert events.index("heading") < events.index("Will create a single AAC/M4B file:")
    assert events.index("Will create a single AAC/M4B file:") < events.index("confirm")
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

    print_screen_heading.assert_called_once_with(
        state, "Start", subheading=concat_menu.make_output_files_subheading
    )
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


def test_pause_timing_rules_menu_offers_limit_silence_gaps() -> None:
    # The setting lives in the "Pause timing rules" submenu; its label reflects the current value
    project = Project.model_validate({"limit_silence_gaps": True, "limit_silence_gaps_duration": 0.4})
    state = cast(State, SimpleNamespace(project=project))

    with patch("tts_audiobook_tool.menus.concat_menu.MenuUtil.menu") as menu:
        ConcatMenu.prosody_modulation_menu(state)
        items = menu.call_args.kwargs["items"](state)

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


def test_concat_menu_offers_room_tone_after_room_reverb_stereo_effect() -> None:
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
    assert labels[index - 1].startswith("Room reverb stereo effect")


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


def test_concat_menu_offers_room_reverb_stereo_after_generative_upsampling() -> None:
    project = Project()
    state = cast(State, SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="128k")))

    with patch("tts_audiobook_tool.menus.concat_menu.MenuUtil.menu") as menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
        return_value=[],
    ), patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
        ConcatMenu.menu(state)
        items = menu.call_args.args[2](state)

    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    index = labels.index("Room reverb stereo effect (currently: Centered (no spread))")
    assert labels[index - 1].startswith("Generative upsampling")
    # Room reverb itself moved to the per-voice settings
    assert not [label for label in labels if label.startswith("Room reverb (")]


def test_room_reverb_stereo_item_always_shows() -> None:
    project = Project()
    state = cast(State, SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="128k")))

    def labels() -> list[str]:
        with patch("tts_audiobook_tool.menus.concat_menu.MenuUtil.menu") as menu, patch(
            "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
            return_value=[],
        ), patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
            ConcatMenu.menu(state)
            items = menu.call_args.args[2](state)
        return [strip_ansi_codes(get_string_from(state, item.label)) for item in items]

    # Shown whether or not the voice in use has reverb on
    assert "Room reverb stereo effect (currently: Centered (no spread))" in labels()
    project.voice_references = [Voice("v.flac", room_reverb=RoomReverb.SMALL_DRR22)]
    assert "Room reverb stereo effect (currently: Centered (no spread))" in labels()


def test_room_reverb_stereo_menu_writes_project_setting() -> None:
    project = Project()
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(concat_menu.MenuUtil, "options_menu") as options_menu, patch.object(
        Project, "save"
    ) as save, patch.object(concat_menu, "print_feedback"):
        ConcatMenu.room_reverb_stereo_menu(state)
        kwargs = options_menu.call_args.kwargs
        assert kwargs["heading_text"] == "Room reverb stereo effect"
        assert kwargs["labels"] == ["Centered (no spread)", "Wide", "Wider"]
        assert kwargs["default_value"] is RoomReverbStereo.MONO

        kwargs["on_select"](RoomReverbStereo.ALL_W100)

    assert project.room_reverb_stereo is RoomReverbStereo.ALL_W100
    save.assert_called_once()
