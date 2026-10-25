from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.menus.text_menu as text_menu_module
from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.constants import COL_DIM
from tts_audiobook_tool.constants_hints import HINT_DIALOG_VOICE, HINT_TOLERANCE_FIRST_CLASS
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.text_menu import TextMenu, on_select_import
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
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
        dir_path="/tmp/project",
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


def make_dialog_import_state(
    dialog_segmentation: bool,
    dialog_line_count: int,
) -> tuple[State, SimpleNamespace]:
    """A fake project whose imported text has the given number of dialog lines."""
    phrase_groups = [
        SimpleNamespace(voice_index=0, is_dialog=False),
        *[SimpleNamespace(voice_index=0, is_dialog=True) for _ in range(dialog_line_count)],
    ]
    project = SimpleNamespace(
        sound_segments=SimpleNamespace(num_generated=lambda: 0, delete_all=lambda: None),
        max_words=80,
        segmentation_strategy=SegmentationStrategy.MULTI_SENTENCE,
        language_code="fr",
        dialog_segmentation=dialog_segmentation,
        dir_path="/tmp/project",
        markers=set(),
        word_substitutions={},
        has_multiple_book_sections=lambda: False,
        phrase_groups=phrase_groups,
        save_calls=[],
        book=SimpleNamespace(segmentation_settings=SimpleNamespace(
            language_code="fr",
            max_words_per_segment=80,
            strategy=SegmentationStrategy.MULTI_SENTENCE,
            dialog_segmentation=dialog_segmentation,
        )),
    )
    project.save = lambda: project.save_calls.append("project") or ""
    state = cast(State, SimpleNamespace(project=project, prefs=object()))
    return state, project


def run_dialog_import(
    monkeypatch, state: State, answers: list[bool],
) -> dict[str, list]:
    """Run the manual import, answering yes/no prompts in order; records output."""
    record: dict[str, list] = {"printed": [], "prompts": [], "hints": [], "book_saves": []}
    project = cast(SimpleNamespace, state.project)
    remaining = list(answers)

    def fake_confirm(message: str) -> bool:
        record["prompts"].append(strip_ansi_codes(message))
        return remaining.pop(0)

    monkeypatch.setattr(text_menu_module.ask, "ask_confirm", fake_confirm)
    monkeypatch.setattr(text_menu_module, "printt", lambda *args: record["printed"].extend(str(a) for a in args))
    monkeypatch.setattr(text_menu_module.MenuUtil, "print_screen_heading", lambda *args, **kwargs: None)
    monkeypatch.setattr(text_menu_module.ask, "ask_enter_to_continue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.hints, "show_hint_if_necessary", lambda *args, **kwargs: record["hints"].append(args[1]))
    monkeypatch.setattr(text_menu_module.ProjectTextIOUtil, "set_phrase_groups_and_save", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_menu_module.ProjectTextIOUtil, "save_book", lambda project: record["book_saves"].append(project) or "")
    monkeypatch.setattr(
        text_menu_module.ask_phrase_groups, "get_from_std_in", lambda *args, **kwargs: (project.phrase_groups, "Raw text"))

    on_select_import(state, MenuItem("Import", lambda *_: None, data="manual"))
    assert not remaining
    return record


@pytest.mark.parametrize(
    ("dialog_segmentation", "dialog_line_count"),
    [(False, 0), (True, 0), (True, 2)],
)
def test_import_summary_reports_dialog_lines_and_prompts_only_when_found(
    monkeypatch, dialog_segmentation: bool, dialog_line_count: int,
) -> None:
    # Dialog is no longer assigned a voice on import: the summary reports the
    # detected lines, and the voice-2 assignment is an opt-in prompt (declined
    # here, which leaves voices as they were).
    from tts_audiobook_tool.constants import COL_ACCENT

    state, project = make_dialog_import_state(dialog_segmentation, dialog_line_count)
    record = run_dialog_import(monkeypatch, state, [False] if dialog_line_count else [])

    assert f"- Dialog segmentation: {COL_ACCENT}{dialog_segmentation}" in record["printed"]
    detected = [line for line in record["printed"] if line.startswith("  Dialog lines detected:")]
    assert detected == ([f"  Dialog lines detected: {COL_ACCENT}{dialog_line_count}"] if dialog_segmentation else [])
    if dialog_line_count:
        assert record["prompts"][0].startswith(f"Assign the {dialog_line_count} dialog lines to voice sample 2?")
    assert [group.voice_index for group in project.phrase_groups] == [0] * (1 + dialog_line_count)
    assert record["book_saves"] == [] and project.save_calls == []
    assert (HINT_DIALOG_VOICE in record["hints"]) is bool(dialog_line_count)


def test_accepting_dialog_voice_prompt_assigns_voice_2(monkeypatch) -> None:
    # Accepting assigns only the dialog lines to voice sample 2 (index 1) and
    # saves the text; no further prompt follows.
    state, project = make_dialog_import_state(True, 2)
    record = run_dialog_import(monkeypatch, state, [True])

    assert [group.voice_index for group in project.phrase_groups] == [0, 1, 1]
    assert record["book_saves"] == [project]
    assert len(record["prompts"]) == 1
    assert project.save_calls == []


def test_dialog_voice_prompt_rolls_back_when_saving_fails(monkeypatch) -> None:
    # In-memory voices must match the file on disk after a failed save.
    state, project = make_dialog_import_state(True, 1)
    errors: list[str] = []
    monkeypatch.setattr(text_menu_module.ask, "ask_error", errors.append)
    monkeypatch.setattr(text_menu_module.ProjectTextIOUtil, "save_book", lambda project: "disk full")
    monkeypatch.setattr(text_menu_module.ask, "ask_confirm", lambda message: True)
    monkeypatch.setattr(text_menu_module, "printt", lambda *args, **kwargs: None)

    text_menu_module.ask_assign_dialog_voice(state, 1)

    assert [group.voice_index for group in project.phrase_groups] == [0, 0]
    assert errors == ["disk full"]


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
        dir_path="/tmp/project",
        markers=markers if markers is not None else set(),
        word_substitutions=substitutions if substitutions is not None else {},
        has_multiple_book_sections=lambda: multi_section,
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
        lambda *args, **kwargs: ([SimpleNamespace(voice_index=0)], "Raw text"),
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
        ),
    )
    on_select_import(state, MenuItem("Import", lambda *_: None, data="manual"))

    return {
        "confirm_calls": confirm_calls,
        "delete_calls": delete_calls,
        "commit_calls": commit_calls,
    }


def test_text_import_confirmation_notes_kept_substitutions(monkeypatch) -> None:
    # Substitutions are not invalidated, but get a dim note when the prompt shows anyway
    project = make_import_project(
        num_files=0,
        markers={1, 2},
        substitutions={"Ariekei": "AriaKay"},
    )

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    assert len(captured["confirm_calls"]) == 1
    raw_message = captured["confirm_calls"][0]
    assert f"{COL_DIM}Note: Word substitutions (1 item) will be kept." in raw_message
    assert strip_ansi_codes(raw_message) == (
        "Replacing project text will invalidate the following:\n"
        "- Section markers (2 items)\n"
        "Note: Word substitutions (1 item) will be kept.\n"
        "\n"
        "Are you sure? "
    )
    assert captured["delete_calls"] == [True]
    assert len(captured["commit_calls"]) == 1


def test_text_import_confirmation_lists_all_invalidated_categories(monkeypatch) -> None:
    project = make_import_project(num_files=3, markers={1})

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    message = strip_ansi_codes(captured["confirm_calls"][0])
    assert message == (
        "Replacing project text will invalidate the following:\n"
        "- Generated sound segments (3 files, deleted from disk)\n"
        "- Section markers (1 item)\n"
        "\n"
        "Are you sure? "
    )
    assert captured["delete_calls"] == [True]


def test_text_import_does_not_confirm_when_only_substitutions_exist(monkeypatch) -> None:
    # Substitutions alone don't warrant a prompt since nothing would be invalidated
    project = make_import_project(substitutions={"a": "b", "c": "d"})

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    assert captured["confirm_calls"] == []
    assert len(captured["commit_calls"]) == 1


def test_text_import_confirmation_uses_split_points_label_for_multi_section(
    monkeypatch,
) -> None:
    project = make_import_project(num_files=0, markers={1, 2}, multi_section=True)

    captured = run_manual_import(monkeypatch, project, confirm_answer=True)

    assert "- Split points (2 items)" in strip_ansi_codes(captured["confirm_calls"][0])


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
