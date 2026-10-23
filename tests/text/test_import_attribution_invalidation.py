"""Text replacement owns invalidation timing; the helper owns range validity."""
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
def test_import_filters_only_changed_ranges_and_preserves_character_data(saved_project, source):
    project, results = saved_project
    incoming = make_book([('“One.”', Reason.PHRASE, DIALOG), ('“Changed.”', Reason.PHRASE, DIALOG)])
    assert import_book(project, incoming, source) == ""
    assert SpeakerAssignFileUtil.load_attributions_file(project.dir_path) == AttributionsFile((results.ranges[0],))
    for path in (SpeakerAssignFileUtil.roster_path(project.dir_path),
                 SpeakerAssignFileUtil.voices_path(project.dir_path)):
        assert Path(path).read_text(encoding="utf-8") == '{"sentinel": true}'


@pytest.mark.parametrize("failure", ["book", "project", "raw", "plan", "cleanup"])
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
    elif failure == "plan":
        monkeypatch.setattr(text_menu.AttributionInvalidation, "plan", lambda *_: "plan failed")
    else:
        monkeypatch.setattr(SpeakerAssignFileUtil, "save_attributions_file", lambda *_: "cleanup failed")
    old_book = project.book
    prefixes = {
        "project": "Project text was saved, but project metadata could not be saved: ",
        "raw": "Project text was saved, but the raw source text could not be saved: ",
        "cleanup": "Project text was saved, but outdated dialog assignments could not be cleared: ",
    }
    assert import_book(project, incoming) == prefixes.get(failure, "") + f"{failure} failed"
    assert SpeakerAssignFileUtil.load_attributions_file(project.dir_path) == results
    if failure in ("plan", "book"):
        assert project.book is old_book


@pytest.mark.parametrize("source", ["text", "manual", "epub"])
@pytest.mark.parametrize("changed,accept,save_error", [
    (True, False, ""), (True, True, ""), (False, True, ""),
    (True, True, "save failed"),
])
def test_import_confirmation_uses_incoming_text_once(saved_project, monkeypatch, source, changed, accept, save_error):
    # Confirmation is based on incoming content; errors never report import success.
    project, results = saved_project
    incoming = make_book([('“New.”' if changed else '“One.”', Reason.PHRASE, DIALOG),
                          ('“Two.”', Reason.PHRASE, DIALOG)])
    prompts, commits, errors, deletes = [], [], [], []
    monkeypatch.setattr(text_menu.MenuUtil, "print_screen_heading", lambda *_: None)
    monkeypatch.setattr(text_menu, "printt", lambda *_: None)
    monkeypatch.setattr(text_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(text_menu.ask, "ask_confirm", lambda msg: prompts.append(msg) or accept)
    monkeypatch.setattr(text_menu.ask, "ask_enter_to_continue", lambda *_: None)
    monkeypatch.setattr(text_menu.hints, "show_hint_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(type(project.sound_segments), "num_generated", lambda _: 0)
    monkeypatch.setattr(type(project.sound_segments), "delete_all", lambda _: deletes.append(True))
    monkeypatch.setattr(text_menu.ask_phrase_groups, "get_from_std_in", lambda *_, **__: (incoming.phrase_groups, "raw"))
    monkeypatch.setattr(text_menu.ask_phrase_groups, "get_from_text_file", lambda *_, **__: (incoming.phrase_groups, "raw", "title"))
    monkeypatch.setattr(text_menu.EpubMenuUtil, "ask_epub_path", lambda _: "book.epub")
    monkeypatch.setattr(text_menu.EpubMenuUtil, "import_epub", lambda **_: EpubImportResult(
        phrase_groups=incoming.phrase_groups, raw_text="raw", section_start_indices=[1], chapters=[]))
    monkeypatch.setattr(text_menu.EpubMenuUtil, "print_import_info", lambda _: None)
    monkeypatch.setattr(text_menu.EpubExtractor, "copy_epub_to_project", lambda *_: "")
    for name in ("set_phrase_groups_and_save", "set_phrase_groups_chapters_and_save"):
        monkeypatch.setattr(ProjectTextIOUtil, name, lambda *_, **kwargs: commits.append(kwargs) or save_error)
    state = cast(State, SimpleNamespace(project=project, prefs=object()))
    text_menu.on_select_import(state, MenuItem("Import", lambda *_: None, data=source))
    assert errors == ([save_error] if save_error else [])
    assert len(prompts) == int(changed)
    if changed:
        assert "1 affected chunk" in prompts[0]
        assert "roster and voice mappings will be kept" in prompts[0]
    assert len(commits) == int(not changed or accept)
    assert len(deletes) == (0 if save_error else len(commits))
    if commits:
        assert commits[0]["invalidation"].removed_count == int(changed)
    # Merely confirming/planning cannot change saved assignments.
    assert SpeakerAssignFileUtil.load_attributions_file(project.dir_path) == results
