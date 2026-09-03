from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.menus.text_menu as text_menu_module
from tts_audiobook_tool.app_types import SegmentationStrategy, VoiceSelectMode
from tts_audiobook_tool.constants_hints import HINT_DIALOG_VOICE, HINT_TOLERANCE_FIRST_CLASS
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.text_menu import TextMenu, on_set_text
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppUnavailable,
    EditorSaveFailed,
    EditorSaved,
)


def make_state() -> State:
    return cast(State, SimpleNamespace(project=object()))


def stub_editor_construction(monkeypatch) -> None:
    monkeypatch.setattr(text_menu_module, "TextEditor", lambda _: object())


def test_text_menu_reports_saved_editor_result(monkeypatch) -> None:
    feedback_calls: list[tuple[str, bool, bool]] = []
    stub_editor_construction(monkeypatch)
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaved()),
    )
    monkeypatch.setattr(
        text_menu_module,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (
                message,
                kwargs.get("is_error", False),
                kwargs.get("long_pause", False),
            )
        ),
    )

    TextMenu.edit_text(make_state())

    assert feedback_calls == [("Saved changes", False, True)]


def test_text_menu_reports_save_failure(monkeypatch) -> None:
    feedback_calls: list[str] = []
    stub_editor_construction(monkeypatch)
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaveFailed("Save failed: disk full")),
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_error", feedback_calls.append)

    TextMenu.edit_text(make_state())

    assert feedback_calls == ["Save failed: disk full"]


def test_text_menu_reports_technical_launch_failure(monkeypatch) -> None:
    feedback_calls: list[str] = []
    stub_editor_construction(monkeypatch)
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppUnavailable("Unsupported terminal"),
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_error", feedback_calls.append)

    TextMenu.edit_text(make_state())

    assert feedback_calls == ["Unsupported terminal"]


@pytest.mark.parametrize("source_kind", ["manual", "import", "epub"])
def test_text_import_shows_tolerance_hint_for_first_class_language(
    monkeypatch, source_kind: str
) -> None:
    hint_calls: list[tuple[object, object, bool]] = []
    phrase_groups = [object()]
    segmentation_settings = SimpleNamespace(
        language_code="en",
        max_words_per_segment=40,
        strategy=SegmentationStrategy.SENTENCE,
        dialog_segmentation=False,
    )
    project = SimpleNamespace(
        sound_segments=SimpleNamespace(num_generated=lambda: 0, delete_all=lambda: None),
        max_words=40,
        segmentation_strategy=SegmentationStrategy.SENTENCE,
        language_code="en",
        dialog_segmentation=False,
        dir_path="/tmp/project",
        book=SimpleNamespace(segmentation_settings=segmentation_settings),
    )
    prefs = object()
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=prefs,
            real_time=SimpleNamespace(custom_phrase_groups=[], project_text_line_range=(1, 2)),
        ),
    )

    monkeypatch.setattr(text_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints,
        "show_hint_if_necessary",
        lambda actual_prefs, hint, and_prompt=False: hint_calls.append(
            (actual_prefs, hint, and_prompt)
        ),
    )
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil,
        "set_phrase_groups_and_save",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil,
        "set_phrase_groups_chapters_and_save",
        lambda *args, **kwargs: None,
    )

    if source_kind == "manual":
        monkeypatch.setattr(
            text_menu_module.ask_phrase_groups,
            "get_from_std_in",
            lambda *args, **kwargs: (phrase_groups, "Raw text"),
        )
    elif source_kind == "import":
        monkeypatch.setattr(
            text_menu_module.ask_phrase_groups,
            "get_from_text_file",
            lambda *args, **kwargs: (phrase_groups, "Raw text", "Title"),
        )
    else:
        epub_result = EpubImportResult(
            phrase_groups=cast(list, phrase_groups),
            raw_text="Raw text",
            section_start_indices=[0],
            chapters=[],
            book_title="Title",
        )
        monkeypatch.setattr(text_menu_module.EpubMenuUtil, "ask_epub_path", lambda _: "/tmp/book.epub")
        monkeypatch.setattr(
            text_menu_module.EpubMenuUtil,
            "import_epub",
            lambda **kwargs: epub_result,
        )
        monkeypatch.setattr(text_menu_module.EpubMenuUtil, "print_import_info", lambda _: None)
        monkeypatch.setattr(
            text_menu_module.EpubExtractor,
            "copy_epub_to_project",
            lambda *args, **kwargs: "",
        )
        monkeypatch.setattr(text_menu_module.text_util, "make_terminal_hyperlink", lambda *args, **kwargs: "raw.txt")

    on_set_text(state, MenuItem("Import", lambda *_: None, data=source_kind))

    assert hint_calls == [(prefs, HINT_TOLERANCE_FIRST_CLASS, False)]


def test_text_import_does_not_show_tolerance_hint_for_other_language(monkeypatch) -> None:
    hint_calls: list[object] = []
    project = SimpleNamespace(
        sound_segments=SimpleNamespace(num_generated=lambda: 0, delete_all=lambda: None),
        max_words=40,
        segmentation_strategy=SegmentationStrategy.SENTENCE,
        language_code="fr",
        dialog_segmentation=False,
        book=SimpleNamespace(
            segmentation_settings=SimpleNamespace(
                language_code="fr",
                max_words_per_segment=40,
                strategy=SegmentationStrategy.SENTENCE,
                dialog_segmentation=False,
            )
        ),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=object(),
            real_time=SimpleNamespace(custom_phrase_groups=[], project_text_line_range=None),
        ),
    )
    monkeypatch.setattr(
        text_menu_module.ask_phrase_groups,
        "get_from_std_in",
        lambda *args, **kwargs: ([object()], "Raw text"),
    )
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil,
        "set_phrase_groups_and_save",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(text_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints,
        "show_hint_if_necessary",
        lambda *args, **kwargs: hint_calls.append(args),
    )

    on_set_text(state, MenuItem("Import", lambda *_: None, data="manual"))

    assert hint_calls == []


@pytest.mark.parametrize("dialog_segmentation", [True, False])
def test_text_import_shows_dialog_voice_hint_when_dialog_segmentation_enabled(
    monkeypatch, dialog_segmentation: bool
) -> None:
    hint_calls: list[object] = []
    project = SimpleNamespace(
        sound_segments=SimpleNamespace(num_generated=lambda: 0, delete_all=lambda: None),
        max_words=40,
        segmentation_strategy=SegmentationStrategy.SENTENCE,
        language_code="fr",
        dialog_segmentation=dialog_segmentation,
        book=SimpleNamespace(
            segmentation_settings=SimpleNamespace(
                language_code="fr",
                max_words_per_segment=40,
                strategy=SegmentationStrategy.SENTENCE,
                dialog_segmentation=dialog_segmentation,
            )
        ),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=object(),
            real_time=SimpleNamespace(custom_phrase_groups=[], project_text_line_range=None),
        ),
    )
    monkeypatch.setattr(
        text_menu_module.ask_phrase_groups,
        "get_from_std_in",
        lambda *args, **kwargs: ([SimpleNamespace(voice_index=-1)], "Raw text"),
    )
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil,
        "set_phrase_groups_and_save",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(text_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints,
        "show_hint_if_necessary",
        lambda *args, **kwargs: hint_calls.append(args),
    )

    on_set_text(state, MenuItem("Import", lambda *_: None, data="manual"))

    if dialog_segmentation:
        assert hint_calls == [(state.prefs, HINT_DIALOG_VOICE)]
    else:
        assert hint_calls == []


@pytest.mark.parametrize(
    (
        "dialog_segmentation",
        "voice_select_mode",
        "dialog_segment_count",
        "expect_inactive_note",
    ),
    [
        (False, VoiceSelectMode.AUTO_ADVANCE, 0, False),
        (True, VoiceSelectMode.AUTO_ADVANCE, 1, True),
        (True, VoiceSelectMode.USER_DEFINED, 1, False),
        (True, VoiceSelectMode.AUTO_ADVANCE, 0, False),
    ],
)
def test_import_summary_reports_dialog_preassignments_without_changing_mode(
    monkeypatch,
    dialog_segmentation: bool,
    voice_select_mode: VoiceSelectMode,
    dialog_segment_count: int,
    expect_inactive_note: bool,
) -> None:
    from tts_audiobook_tool.constants import COL_ACCENT

    phrase_groups = [
        SimpleNamespace(voice_index=-1),
        *[
            SimpleNamespace(voice_index=1)
            for _ in range(dialog_segment_count)
        ],
    ]
    segmentation_settings = SimpleNamespace(
        language_code="en",
        max_words_per_segment=80,
        strategy=SegmentationStrategy.MULTI_SENTENCE,
        dialog_segmentation=dialog_segmentation,
    )
    project = SimpleNamespace(
        sound_segments=SimpleNamespace(num_generated=lambda: 0, delete_all=lambda: None),
        max_words=80,
        segmentation_strategy=SegmentationStrategy.MULTI_SENTENCE,
        language_code="en",
        dialog_segmentation=dialog_segmentation,
        voice_select_mode=voice_select_mode,
        dir_path="/tmp/project",
        book=SimpleNamespace(segmentation_settings=segmentation_settings),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=object(),
            real_time=SimpleNamespace(custom_phrase_groups=[], project_text_line_range=None),
        ),
    )

    printed: list[str] = []
    monkeypatch.setattr(text_menu_module, "printt", lambda *args: printed.extend(str(a) for a in args))
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints,
        "show_hint_if_necessary",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil,
        "set_phrase_groups_and_save",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        text_menu_module.ask_phrase_groups,
        "get_from_std_in",
        lambda *args, **kwargs: (phrase_groups, "Raw text"),
    )

    on_set_text(state, MenuItem("Import", lambda *_: None, data="manual"))

    expected = f"- Dialog segmentation: {COL_ACCENT}{dialog_segmentation}"
    assert expected in printed
    assert printed.index(expected) > printed.index(
        f"- Text segmenter strategy: {COL_ACCENT}Multiple sentences"
    )

    assignment_lines = [
        line
        for line in printed
        if line.startswith("  Text segments preassigned to voice sample 2:")
    ]
    assert assignment_lines == (
        [
            f"  Text segments preassigned to voice sample 2: "
            f"{COL_ACCENT}{dialog_segment_count}"
        ]
        if dialog_segmentation
        else []
    )
    assert any("Voice selection mode must be set" in line for line in printed) is expect_inactive_note
    assert project.voice_select_mode is voice_select_mode
