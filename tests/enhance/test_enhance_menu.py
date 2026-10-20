from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.enhance import enhance_flow, enhance_menu
from tts_audiobook_tool.enhance.enhance_artifacts import (
    EnhanceArtifacts,
    EnhanceState,
)
from tts_audiobook_tool.enhance.enhance_menu import (
    INTRO,
    audio_status,
    build_items,
    output_status,
    output_suffix,
    text_status,
)


def snapshot(tmp_path: Path, *, source_kind: str | None = None, audio=True, transcription=False, alignment=False, output=False, count=0):
    artifacts = EnhanceArtifacts.from_audio_path(tmp_path / "directory" / "very-long-audiobook-filename.m4b")
    book = None
    if source_kind is not None:
        book = Book(
            sections=[BookSection([])],
            text_source_kind=source_kind,
            audio_source_kind="pre_existing",
        )
    return EnhanceState(
        artifacts=artifacts,
        has_audio_path=True,
        audio_exists=audio,
        book=book,
        book_error="",
        transcription_exists=transcription,
        transcription_valid=transcription,
        transcription_error="",
        timed_phrases_exists=alignment,
        timed_phrases_valid=alignment,
        timed_phrases_error="",
        expected_output_path=artifacts.flat_output_path if source_kind == "plain_text" else artifacts.epub_output_path,
        output_exists=output,
        output_valid=output,
        output_error="",
        output_phrase_count=count,
        misalignment_count=count,
    )


def stripped_labels(items):
    return [text_util.strip_ansi_codes(item.label) for item in items]


def test_audio_status_is_empty_or_current_path_hyperlinked(tmp_path: Path) -> None:
    empty = replace(snapshot(tmp_path), artifacts=None, has_audio_path=False, audio_exists=False)
    assert audio_status(empty) == ""
    current = audio_status(snapshot(tmp_path))
    stripped = text_util.strip_ansi_codes(current)
    assert stripped.startswith("(currently: ")
    assert stripped.endswith("very-long-audiobook-filename.m4b)")
    assert "]8;;file://" in current  # terminal hyperlink wraps the path


def test_text_status_counts_book_lines(tmp_path: Path) -> None:
    assert text_status(snapshot(tmp_path)) == ""
    book = snapshot(tmp_path, source_kind="plain_text").book
    phrases = [Phrase(f"line {i}", Reason.SENTENCE) for i in range(3)]
    populated = replace(snapshot(tmp_path, source_kind="plain_text"), book=replace(book, sections=[BookSection([PhraseGroup(phrases)])]))
    assert text_util.strip_ansi_codes(text_status(populated)) == "(currently: 3 lines)"


def test_output_status_hyperlinks_existing_file(tmp_path: Path) -> None:
    assert output_status(snapshot(tmp_path, source_kind="plain_text")) == ""
    current = output_status(snapshot(tmp_path, source_kind="plain_text", output=True))
    stripped = text_util.strip_ansi_codes(current)
    assert stripped.startswith("(")
    assert stripped.endswith(".abr.m4a)")
    assert "]8;;file://" in current  # terminal hyperlink wraps the path


def test_transcribe_align_and_create_labels_have_independent_status(tmp_path: Path) -> None:
    state = snapshot(tmp_path, source_kind="plain_text", transcription=True)
    labels = stripped_labels(build_items(SimpleNamespace(), state))
    assert labels[1] == "✅ Transcribe"
    assert labels[2].startswith("✅ Enter text or EPUB file path")
    assert labels[3] == "⬜ Align source text with transcription"
    assert labels[4] == '⬜ Create the ".abr.m4a" file'

    aligned = snapshot(tmp_path, source_kind="epub", alignment=True)
    labels = stripped_labels(build_items(SimpleNamespace(), aligned))
    assert labels[1] == "⬜ Transcribe"
    assert labels[3] == "✅ Align source text with transcription"
    assert labels[4] == '⬜ Create the ".abr.m4b" file'


def test_stale_alignment_without_book_shows_align_unchecked(tmp_path: Path) -> None:
    state = snapshot(tmp_path, alignment=True)  # timed_phrases valid, book deleted

    labels = stripped_labels(build_items(SimpleNamespace(), state))

    assert labels[2] == "⬜ Enter text or EPUB file path"
    assert labels[3] == "⬜ Align source text with transcription"
    # The audio track is unaffected.
    assert labels[1] == "⬜ Transcribe"


def test_dynamic_suffix_depends_on_book_source(tmp_path: Path) -> None:
    assert output_suffix(snapshot(tmp_path)) == ".abr.m4b"
    assert output_suffix(snapshot(tmp_path, source_kind="epub")) == ".abr.m4b"
    assert output_suffix(snapshot(tmp_path, source_kind="plain_text")) == ".abr.m4a"


def test_hotkeys_are_auto_assigned_and_optional_spacing(tmp_path: Path) -> None:
    app_state = SimpleNamespace()

    no_selection = replace(snapshot(tmp_path), artifacts=None, has_audio_path=False, audio_exists=False)
    items = build_items(app_state, no_selection)
    assert all(item.hotkey == "" for item in items)
    # The two input tracks are visually grouped, plus a gap before "Create".
    assert [item.blank_line_before for item in items] == [False, False, True, False, True]

    selected = build_items(app_state, snapshot(tmp_path))
    assert all(item.hotkey == "" for item in selected)
    assert [item.blank_line_before for item in selected] == [False, False, True, False, True, True]

    completed = build_items(app_state, snapshot(tmp_path, output=True, count=1))
    assert all(item.hotkey == "" for item in completed)
    assert [item.blank_line_before for item in completed] == [False, False, True, False, True, True, False]
    labels = stripped_labels(completed)
    assert labels[5] == "Review unmatched lines (1 of 1 lines)"


def test_review_label_uses_output_total_not_current_book_lines(tmp_path: Path) -> None:
    current = snapshot(tmp_path, source_kind="plain_text", output=True, count=1)
    current = replace(current, output_phrase_count=2)

    labels = stripped_labels(build_items(SimpleNamespace(), current))

    assert labels[5] == "Review unmatched lines (1 of 2 lines)"


def test_empty_state_item_labels_have_no_status_copy(tmp_path: Path) -> None:
    state = replace(snapshot(tmp_path), artifacts=None, has_audio_path=False, audio_exists=False)
    assert stripped_labels(build_items(SimpleNamespace(), state)) == [
        "⬜ Enter source audiobook file path",
        "⬜ Transcribe",
        "⬜ Enter text or EPUB file path",
        "⬜ Align source text with transcription",
        '⬜ Create the ".abr.m4b" file',
    ]


def test_menu_loads_one_snapshot_per_render(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "book.m4b"
    audio.write_bytes(b"")
    app_state = SimpleNamespace(prefs=SimpleNamespace(enhance_audio_path=str(audio)))
    current = snapshot(tmp_path)
    loads: list[str] = []

    def load(path):
        loads.append(path)
        return current

    def render(_state, _heading, item_maker, **kwargs):
        assert kwargs["subheading"] == INTRO
        assert all(item.hotkey == "" for item in item_maker(app_state))

    monkeypatch.setattr(enhance_menu, "make_enhance_state_cached", load)
    monkeypatch.setattr(enhance_menu.MenuUtil, "menu", render)

    enhance_menu.menu(app_state)

    assert loads == [str(audio)]


def test_menu_clears_selection_when_audio_file_missing(tmp_path: Path, monkeypatch) -> None:
    saved: list[bool] = []
    app_state = SimpleNamespace(
        prefs=SimpleNamespace(
            enhance_audio_path=str(tmp_path / "missing.m4b"),
            save=lambda: saved.append(True),
        )
    )
    monkeypatch.setattr(enhance_menu.MenuUtil, "menu", lambda *args, **kwargs: None)

    enhance_menu.menu(app_state)

    assert app_state.prefs.enhance_audio_path == ""
    assert saved == [True]


def test_handlers_call_corresponding_flow_operation(tmp_path: Path, monkeypatch) -> None:
    app_state = SimpleNamespace()
    called: list[str] = []
    names = [
        "select_audio",
        "transcribe",
        "select_text",
        "align_source_text",
        "create_output",
        "review_discontinuities",
        "clear_selection",
    ]
    for name in names:
        monkeypatch.setattr(
            enhance_flow,
            name,
            lambda state, operation=name: called.append(operation),
            raising=False,
        )

    items = build_items(app_state, snapshot(tmp_path, output=True))
    for item in items:
        item.handler(app_state, item)

    assert called == names
