from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.menus.text_menu as text_menu_module
from tts_audiobook_tool.app_types import SegmentationStrategy, VoiceSelectMode
from tts_audiobook_tool.constants_hints import HINT_DIALOG_VOICE, HINT_TOLERANCE_FIRST_CLASS
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.text_menu import TextMenu, on_select_import
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    ContentAppUnavailable,
    EditorSaveFailed,
    EditorSaved,
)
from tts_audiobook_tool.textual.word_substitutions_app import (
    WordSubstitutionPreviewRequested,
)


def make_state() -> State:
    return cast(State, SimpleNamespace(project=object()))


def stub_editor_construction(monkeypatch) -> None:
    monkeypatch.setattr(text_menu_module, "TextEditor", lambda _: object())


def test_text_menu_saved_editor_result_does_not_report_error(monkeypatch) -> None:
    errors: list[str] = []
    stub_editor_construction(monkeypatch)
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaved()),
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_error", errors.append)

    TextMenu.edit_text(make_state())

    assert errors == []


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


@pytest.mark.parametrize("source_kind", ["manual", "text", "epub"])
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
        markers=set(),
        word_substitutions={},
        has_multiple_book_sections=lambda: False,
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
    monkeypatch.setattr(
        text_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None
    )
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
    elif source_kind == "text":
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

    on_select_import(state, MenuItem("Import", lambda *_: None, data=source_kind))

    assert hint_calls == [(prefs, HINT_TOLERANCE_FIRST_CLASS, False)]


def test_text_import_does_not_show_tolerance_hint_for_other_language(monkeypatch) -> None:
    hint_calls: list[object] = []
    project = SimpleNamespace(
        sound_segments=SimpleNamespace(num_generated=lambda: 0, delete_all=lambda: None),
        max_words=40,
        segmentation_strategy=SegmentationStrategy.SENTENCE,
        language_code="fr",
        dialog_segmentation=False,
        markers=set(),
        word_substitutions={},
        has_multiple_book_sections=lambda: False,
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
    monkeypatch.setattr(
        text_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints,
        "show_hint_if_necessary",
        lambda *args, **kwargs: hint_calls.append(args),
    )

    on_select_import(state, MenuItem("Import", lambda *_: None, data="manual"))

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
        markers=set(),
        word_substitutions={},
        has_multiple_book_sections=lambda: False,
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
    monkeypatch.setattr(
        text_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints,
        "show_hint_if_necessary",
        lambda *args, **kwargs: hint_calls.append(args),
    )

    on_select_import(state, MenuItem("Import", lambda *_: None, data="manual"))

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
        markers=set(),
        word_substitutions={},
        has_multiple_book_sections=lambda: False,
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
    monkeypatch.setattr(
        text_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None
    )
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

    on_select_import(state, MenuItem("Import", lambda *_: None, data="manual"))

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


def make_import_project(
    *,
    num_files: int = 0,
    markers: set[int] | None = None,
    substitutions: dict[str, str] | None = None,
    multi_section: bool = False,
) -> SimpleNamespace:
    """Build a fake project for exercising the text-import confirmation."""
    return SimpleNamespace(
        sound_segments=SimpleNamespace(
            num_generated=lambda: num_files, delete_all=lambda: None
        ),
        max_words=40,
        segmentation_strategy=SegmentationStrategy.SENTENCE,
        language_code="en",
        dialog_segmentation=False,
        markers=markers if markers is not None else set(),
        word_substitutions=substitutions if substitutions is not None else {},
        has_multiple_book_sections=lambda: multi_section,
        voice_select_mode=VoiceSelectMode.AUTO_ADVANCE,
        book=SimpleNamespace(
            segmentation_settings=SimpleNamespace(
                language_code="en",
                max_words_per_segment=40,
                strategy=SegmentationStrategy.SENTENCE,
                dialog_segmentation=False,
            )
        ),
    )


def run_manual_import(
    monkeypatch, project: SimpleNamespace, *, confirm_answer: bool
) -> dict:
    """Run the manual-import route, recording confirmation and commit calls."""
    confirm_calls: list[str] = []
    delete_calls: list[bool] = []
    commit_calls: list[dict] = []

    def fake_confirm(message: str) -> bool:
        confirm_calls.append(message)
        return confirm_answer

    project.sound_segments.delete_all = lambda: delete_calls.append(True)
    monkeypatch.setattr(text_menu_module.ask, "ask_confirm", fake_confirm)
    monkeypatch.setattr(
        text_menu_module.ask_phrase_groups,
        "get_from_std_in",
        lambda *args, **kwargs: ([SimpleNamespace(voice_index=-1)], "Raw text"),
    )
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil,
        "set_phrase_groups_and_save",
        lambda project, **kwargs: commit_calls.append(kwargs),
    )
    monkeypatch.setattr(text_menu_module, "printt", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        text_menu_module.hints, "show_hint_if_necessary", lambda *args, **kwargs: None
    )

    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=object(),
            real_time=SimpleNamespace(custom_phrase_groups=[], project_text_line_range=None),
        ),
    )
    on_select_import(state, MenuItem("Import", lambda *_: None, data="manual"))

    return {
        "confirm_calls": confirm_calls,
        "delete_calls": delete_calls,
        "commit_calls": commit_calls,
    }


def test_text_import_confirms_when_only_markers_and_substitutions_exist(monkeypatch) -> None:
    project = make_import_project(
        num_files=0,
        markers={1, 2},
        substitutions={"Ariekei": "AriaKay"},
    )

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    assert len(captured["confirm_calls"]) == 1
    message = strip_ansi_codes(captured["confirm_calls"][0])
    assert "2 section markers" in message
    assert "1 word substitution" in message
    assert "generated sound segment" not in message
    assert captured["delete_calls"] == [True]
    assert len(captured["commit_calls"]) == 1


def test_text_import_confirmation_names_all_invalidated_categories(monkeypatch) -> None:
    project = make_import_project(
        num_files=3,
        markers={1},
        substitutions={"a": "b", "c": "d"},
    )

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    message = strip_ansi_codes(captured["confirm_calls"][0])
    assert "3 generated sound segment files" in message
    assert "1 section marker" in message
    assert "2 word substitutions" in message
    assert captured["delete_calls"] == [True]


def test_text_import_confirmation_uses_split_points_label_for_multi_section(
    monkeypatch,
) -> None:
    project = make_import_project(num_files=0, markers={1, 2}, multi_section=True)

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    assert "2 split points" in strip_ansi_codes(captured["confirm_calls"][0])


def test_text_import_declined_confirmation_skips_import(monkeypatch) -> None:
    project = make_import_project(num_files=2)

    captured = run_manual_import(monkeypatch, project, confirm_answer=False)

    assert len(captured["confirm_calls"]) == 1
    assert captured["delete_calls"] == []
    assert captured["commit_calls"] == []


def test_text_import_does_not_confirm_when_nothing_to_invalidate(monkeypatch) -> None:
    project = make_import_project()

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    assert captured["confirm_calls"] == []
    assert len(captured["commit_calls"]) == 1


def make_substitutions_state(values: dict[str, str]) -> State:
    project = Project.model_validate({"word_substitutions": values})
    return cast(State, SimpleNamespace(project=project))


def make_phrase_group(words: list[str]) -> SimpleNamespace:
    return SimpleNamespace(phrases=[SimpleNamespace(words=words)])


def word_substitutions_menu_items(monkeypatch, state: State) -> list[MenuItem]:
    captured: dict = {}

    def fake_menu(*args, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(text_menu_module.MenuUtil, "menu", fake_menu)
    TextMenu.word_substitutions_menu(state)
    items_maker = captured["items"]
    return items_maker(state)


def test_word_substitutions_menu_has_single_edit_item(monkeypatch) -> None:
    state = make_substitutions_state({})

    items = word_substitutions_menu_items(monkeypatch, state)

    assert len(items) == 1
    label = strip_ansi_codes(get_string_from(state, items[0].label))
    assert label == "Edit word substitutions (items: 0)"


def test_word_substitutions_menu_label_counts_items(monkeypatch) -> None:
    state = make_substitutions_state({"Ariekei": "AriaKay", "kilohour": "kilo hour"})

    items = word_substitutions_menu_items(monkeypatch, state)

    label = strip_ansi_codes(get_string_from(state, items[0].label))
    assert label == "Edit word substitutions (items: 2)"


def test_word_substitutions_menu_includes_uncommon_words_inspector(monkeypatch) -> None:
    state = make_substitutions_state({})
    state.project.phrase_groups = cast(list, [make_phrase_group(["Ariekei"])])

    items = word_substitutions_menu_items(monkeypatch, state)

    assert len(items) == 2
    label = strip_ansi_codes(get_string_from(state, items[1].label))
    assert label == "Inspect project text for uncommon words"


def test_word_substitutions_menu_omits_inspector_without_phrase_groups(monkeypatch) -> None:
    state = make_substitutions_state({})

    items = word_substitutions_menu_items(monkeypatch, state)

    assert len(items) == 1


def test_word_substitutions_menu_omits_inspector_for_unsupported_language(monkeypatch) -> None:
    state = make_substitutions_state({})
    state.project.phrase_groups = cast(list, [make_phrase_group(["Ariekei"])])
    state.project.language_code = "zz"

    items = word_substitutions_menu_items(monkeypatch, state)

    assert len(items) == 1


@pytest.mark.parametrize("language_code", ["en-US", "EN", "English", "es-419", "spa"])
def test_word_substitutions_menu_inspector_accepts_first_class_language_aliases(
    monkeypatch, language_code: str
) -> None:
    state = make_substitutions_state({})
    state.project.phrase_groups = cast(list, [make_phrase_group(["Ariekei"])])
    state.project.language_code = language_code

    items = word_substitutions_menu_items(monkeypatch, state)

    assert len(items) == 2


def test_inspect_uncommon_words_prints_frequency_and_instances(monkeypatch, capsys) -> None:
    state = make_substitutions_state({})
    state.project.phrase_groups = cast(list, [make_phrase_group(["Ariekei", "Ariekei"])])
    headings: list[dict] = []
    monkeypatch.setattr(
        text_menu_module.MenuUtil,
        "print_screen_heading",
        lambda *a, **k: headings.append({"args": a, "kwargs": k}),
    )
    monkeypatch.setattr(text_menu_module, "printt", lambda *a, **k: None)
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *a, **k: None)
    monkeypatch.setattr(
        text_menu_module.app_text,
        "get_uncommon_words",
        lambda words: [("ariekai", 2, ["Ariekei", "ariekai"])],
    )

    items = word_substitutions_menu_items(monkeypatch, state)
    items[1].handler(state, items[1])

    printed = strip_ansi_codes(capsys.readouterr().out).strip()
    assert printed == "2  Ariekei | ariekai"
    assert headings[0]["args"][1] == "Uncommon words"
    subheading = headings[0]["kwargs"]["subheading"]
    assert "%1" not in subheading
    assert "en" in subheading


def test_inspect_uncommon_words_subheading_uses_normalized_language(monkeypatch) -> None:
    state = make_substitutions_state({})
    state.project.phrase_groups = cast(list, [make_phrase_group(["Ariekei"])])
    state.project.language_code = "en-US"
    headings: list[dict] = []
    monkeypatch.setattr(
        text_menu_module.MenuUtil,
        "print_screen_heading",
        lambda *a, **k: headings.append({"args": a, "kwargs": k}),
    )
    monkeypatch.setattr(text_menu_module, "printt", lambda *a, **k: None)
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *a, **k: None)
    monkeypatch.setattr(text_menu_module.app_text, "get_uncommon_words", lambda words: [])

    items = word_substitutions_menu_items(monkeypatch, state)
    items[1].handler(state, items[1])

    subheading = headings[0]["kwargs"]["subheading"]
    assert "en-US" not in subheading
    assert "en" in subheading


def test_inspect_uncommon_words_reports_none_found(monkeypatch, capsys) -> None:
    state = make_substitutions_state({})
    state.project.phrase_groups = cast(list, [make_phrase_group(["the"])])
    heading_calls: list[str] = []
    monkeypatch.setattr(
        text_menu_module.MenuUtil,
        "print_screen_heading",
        lambda *a, **k: heading_calls.append(a[1] if len(a) > 1 else ""),
    )
    monkeypatch.setattr(text_menu_module, "printt", lambda *a, **k: print(*a, **k))
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *a, **k: None)
    monkeypatch.setattr(text_menu_module.app_text, "get_uncommon_words", lambda words: [])

    items = word_substitutions_menu_items(monkeypatch, state)
    items[1].handler(state, items[1])

    assert heading_calls == ["Uncommon words"]
    assert "None found" in strip_ansi_codes(capsys.readouterr().out)


def test_edit_word_substitutions_previews_then_reopens_with_staged_state(
    monkeypatch,
) -> None:
    state = make_state()
    constructed: list[tuple[State, dict[str, object]]] = []
    run_results = iter(
        [
            ContentAppCompleted(
                WordSubstitutionPreviewRequested(
                    "Ariekei",
                    "AriaKay",
                    (("Ariekei", "AriaKay"), ("other", "value")),
                )
            ),
            ContentAppCompleted(EditorSaved()),
        ]
    )
    preview_calls: list[tuple[State, str]] = []
    sound = object()

    def make_editor(passed_state: State, **kwargs):
        constructed.append((passed_state, kwargs))
        return object()

    monkeypatch.setattr(text_menu_module, "WordSubstitutionsApp", make_editor)
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: next(run_results),
    )
    monkeypatch.setattr(
        text_menu_module,
        "run_tts_preview_app",
        lambda passed_state, prompt: (
            preview_calls.append((passed_state, prompt))
            or SimpleNamespace(completed=True, sound=sound)
        ),
    )

    assert TextMenu.edit_word_substitutions(state) is True
    assert preview_calls == [
        (state, "Original word: Ariekei. Substitute word: AriaKay")
    ]
    assert constructed == [
        (
            state,
            {
                "staged": None,
                "restore_original": None,
                "preview_sound": None,
            },
        ),
        (
            state,
            {
                "staged": {"Ariekei": "AriaKay", "other": "value"},
                "restore_original": "Ariekei",
                "preview_sound": sound,
            },
        ),
    ]


def test_edit_word_substitutions_returns_true_on_saved(monkeypatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(
        text_menu_module, "WordSubstitutionsApp", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaved()),
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_error", errors.append)

    assert TextMenu.edit_word_substitutions(make_state()) is True
    assert errors == []


def test_edit_word_substitutions_reports_save_failure(monkeypatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(
        text_menu_module, "WordSubstitutionsApp", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppCompleted(EditorSaveFailed("Save failed: disk full")),
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_error", errors.append)

    TextMenu.edit_word_substitutions(make_state())

    assert errors == ["Save failed: disk full"]


def test_edit_word_substitutions_reports_launch_failure(monkeypatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(
        text_menu_module, "WordSubstitutionsApp", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        text_menu_module,
        "run_content_textual_app",
        lambda _: ContentAppUnavailable("Unsupported terminal"),
    )
    monkeypatch.setattr(text_menu_module.ask, "ask_error", errors.append)

    TextMenu.edit_word_substitutions(make_state())

    assert errors == ["Unsupported terminal"]
