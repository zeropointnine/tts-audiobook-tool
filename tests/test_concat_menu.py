from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types import ExportType, SectionMarkerMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus import concat_menu
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.menus.menu_util import get_string_from
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

    with patch.object(concat_menu.ask, "ask_confirm", return_value=True), \
        patch.object(concat_menu.OutputRangeInfo, "make_single_info", return_value=SimpleNamespace(num_files_exist=1, num_segments=3)), \
            patch.object(concat_menu.ConcatUtil, "make_files") as make_files_mock, \
            patch.object(concat_menu, "printt"):
        concat_menu.ask_output_indices_and_make(cast(State, state))

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
            patch.object(concat_menu.ConcatUtil, "make_files") as make_files_mock, \
            patch.object(concat_menu, "printt"):
        concat_menu.ask_output_indices_and_make(cast(State, state))

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
        make_items = menu.call_args.args[2]
        items = make_items(state)

    labels = [get_string_from(state, item.label) for item in items]
    assert any(label.startswith("Generative upsampling") for label in labels)


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
    assert not project.use_upsampler
    save.assert_not_called()
    ask_error.assert_called_once_with(
        "LavaSR v2 is not installed; generative upsampling cannot be enabled"
    )
