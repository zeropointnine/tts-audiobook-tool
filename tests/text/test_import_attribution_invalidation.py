"""Text import discards all saved dialog assignments, but only after the text is saved."""
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book
from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.menus import text_menu
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.epub_extractor import EpubImportResult
from tts_audiobook_tool.text_util import strip_ansi_codes


@pytest.fixture
def saved_project(tmp_path):
    book = make_book([('“One.”', Reason.PHRASE, DIALOG), ('“Two.”', Reason.PHRASE, DIALOG)])
    project = Project(dir_path=str(tmp_path), book=book)
    results = AttributionsFile(tuple(
        RangeAttribution(i, i + 1, UtteranceUtil.range_text_hash(book, i, i + 1), ())
        for i in range(2)
    ))
    assert SpeakerAssignFileUtil.save_attributions_file(str(tmp_path), results) == ""
    # These independent artifacts must remain byte-identical, even on text replacement.
    for path in (SpeakerAssignFileUtil.roster_path(str(tmp_path)),
                 SpeakerAssignFileUtil.voices_path(str(tmp_path))):
        Path(path).write_text('{"sentinel": true}', encoding="utf-8")
    return project, results


def import_book(project, incoming, source="plain_text"):
    kwargs = dict(
        phrase_groups=incoming.phrase_groups, strategy=SegmentationStrategy.SENTENCE,
        max_words=40, language_code="en", raw_text="replacement",
    )
    if source == "epub":
        return ProjectTextIOUtil.set_phrase_groups_chapters_and_save(
            project, section_start_indices=[1], section_titles=["A", "B"], **kwargs,
        )
    return ProjectTextIOUtil.set_phrase_groups_and_save(project, text_source_kind=source, **kwargs)


@pytest.mark.parametrize("source", ["plain_text", "manual", "epub"])
def test_import_clears_all_assignments_and_preserves_character_data(saved_project, source):
    # Even the unchanged first passage loses its assignment: import starts over.
    project, _ = saved_project
    incoming = make_book([('“One.”', Reason.PHRASE, DIALOG), ('“Changed.”', Reason.PHRASE, DIALOG)])
    assert import_book(project, incoming, source) == ""
    assert not Path(SpeakerAssignFileUtil.attributions_path(project.dir_path)).exists()
    for path in (SpeakerAssignFileUtil.roster_path(project.dir_path),
                 SpeakerAssignFileUtil.voices_path(project.dir_path)):
        assert Path(path).read_text(encoding="utf-8") == '{"sentinel": true}'


@pytest.mark.parametrize("failure", ["book", "project", "raw", "cleanup"])
def test_import_surfaces_failure_without_early_cleanup(saved_project, monkeypatch, failure):
    # Cleanup is only allowed after every text persistence step has succeeded.
    project, results = saved_project
    incoming = make_book([('“Changed.”', Reason.PHRASE, DIALOG)])
    if failure == "book":
        monkeypatch.setattr(ProjectTextIOUtil, "save_book", lambda _: "book failed")
    elif failure == "project":
        monkeypatch.setattr(Project, "save", lambda _: "project failed")
    elif failure == "raw":
        monkeypatch.setattr(ProjectTextIOUtil, "_save_raw_text", lambda *_: "raw failed")
    else:
        monkeypatch.setattr(SpeakerAssignFileUtil, "delete_attributions_file", lambda *_: "cleanup failed")
    old_book = project.book
    prefixes = {
        "project": "Project text was saved, but project metadata could not be saved: ",
        "raw": "Project text was saved, but the raw source text could not be saved: ",
        "cleanup": "Project text was saved, but saved dialog assignments could not be cleared: ",
    }
    assert import_book(project, incoming) == prefixes.get(failure, "") + f"{failure} failed"
    assert SpeakerAssignFileUtil.load_attributions_file(project.dir_path) == results
    if failure == "book":
        assert project.book is old_book


def run_import(monkeypatch, project, source, accept, save_error=""):
    """Drive on_select_import, recording the order of prompts, input requests and commits."""
    events: list[tuple[str, object]] = []
    incoming = make_book([('“New.”', Reason.PHRASE, DIALOG)])

    def ask_input(result):
        def fn(*_, **__):
            events.append(("input", None))
            return result
        return fn

    def commit(*_, **kwargs):
        events.append(("commit", kwargs))
        return save_error

    monkeypatch.setattr(text_menu.MenuUtil, "print_screen_heading", lambda *_: None)
    monkeypatch.setattr(text_menu, "printt", lambda *_: None)
    monkeypatch.setattr(text_menu.ask, "ask_error", lambda msg: events.append(("error", msg)))
    monkeypatch.setattr(text_menu.ask, "ask_confirm", lambda msg: events.append(("confirm", msg)) or accept)
    monkeypatch.setattr(text_menu.ask, "ask_enter_to_continue", lambda *_: None)
    monkeypatch.setattr(text_menu.hints, "show_hint_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(type(project.sound_segments), "num_generated", lambda _: 0)
    monkeypatch.setattr(type(project.sound_segments), "delete_all", lambda _: events.append(("delete", None)))
    monkeypatch.setattr(text_menu.ask_phrase_groups, "get_from_std_in",
                        ask_input((incoming.phrase_groups, "raw")))
    monkeypatch.setattr(text_menu.ask_phrase_groups, "get_from_text_file",
                        ask_input((incoming.phrase_groups, "raw", "title")))
    monkeypatch.setattr(text_menu.EpubMenuUtil, "ask_epub_path", ask_input("book.epub"))
    monkeypatch.setattr(text_menu.EpubMenuUtil, "import_epub", lambda **_: EpubImportResult(
        phrase_groups=incoming.phrase_groups, raw_text="raw", section_start_indices=[1], chapters=[]))
    monkeypatch.setattr(text_menu.EpubMenuUtil, "print_import_info", lambda _: None)
    monkeypatch.setattr(text_menu.EpubExtractor, "copy_epub_to_project", lambda *_: "")
    for name in ("set_phrase_groups_and_save", "set_phrase_groups_chapters_and_save"):
        monkeypatch.setattr(ProjectTextIOUtil, name, commit)
    state = cast(State, SimpleNamespace(project=project, prefs=object()))
    text_menu.on_select_import(state, MenuItem("Import", lambda *_: None, data=source))
    return events


@pytest.mark.parametrize("source", ["text", "manual", "epub"])
@pytest.mark.parametrize("accept,save_error", [(False, ""), (True, ""), (True, "save failed")])
def test_import_confirms_assignment_loss_before_asking_for_text(
        saved_project, monkeypatch, source, accept, save_error):
    # The prompt precedes the file/text request, counts every saved chunk, and
    # declining never asks for input; errors never report import success.
    project, results = saved_project
    events = run_import(monkeypatch, project, source, accept, save_error)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "confirm"
    message = strip_ansi_codes(cast(str, events[0][1]))
    assert "- Saved dialog assignments (2 chunks)\n" in message
    assert "roster and voice mappings will be preserved" in message
    if not accept:
        assert kinds == ["confirm"]
    elif save_error:
        assert kinds == ["confirm", "input", "commit", "error"]
    else:
        assert kinds == ["confirm", "input", "commit", "delete"]
    # The menu never touches saved assignments itself; the commit helper owns that.
    assert SpeakerAssignFileUtil.load_attributions_file(project.dir_path) == results


def test_import_confirmation_lists_unreadable_assignments_without_count(tmp_path):
    # An unreadable file is cleared on import anyway, so it is listed rather than blocking.
    project = Project(dir_path=str(tmp_path), book=make_book([]))
    path = Path(SpeakerAssignFileUtil.attributions_path(str(tmp_path)))
    path.parent.mkdir()
    path.write_text("{invalid", encoding="utf-8")
    message = strip_ansi_codes(text_menu.make_text_replacement_confirm_message(project))
    assert "- Saved dialog assignments\n" in message


def test_import_without_saved_assignments_does_not_mention_them(tmp_path):
    project = Project(dir_path=str(tmp_path), book=make_book([]))
    assert text_menu.make_text_replacement_confirm_message(project) == ""
