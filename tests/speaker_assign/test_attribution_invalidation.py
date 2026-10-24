from pathlib import Path

import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.attribution_invalidation import AttributionInvalidation
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tests.speaker_assign.speaker_assign_test_support import DIALOG, make_book


def saved_assignments(book) -> AttributionsFile:
    return AttributionsFile(tuple(
        RangeAttribution(i, i + 1, UtteranceUtil.range_text_hash(book, i, i + 1),
                         (UtteranceAttribution((i,), "ALICE"),))
        for i in range(len(book.flat_phrase_groups))
    ))


@pytest.mark.parametrize("mutation, removed", [("edit", 1), ("delete", 2), ("same", 0)])
def test_cleanup_is_previewed_then_persisted_without_touching_configuration(tmp_path, mutation, removed):
    # A local edit retains other chunks; deletion invalidates shifted line ranges.
    book = make_book([(text, Reason.PHRASE_QUOTE_END, DIALOG) for text in ["One.", "Two.", "Three."]])
    saved = saved_assignments(book)
    project_dir = str(tmp_path)
    assert SpeakerAssignFileUtil.save_attributions_file(project_dir, saved) == ""
    configuration_paths = [Path(SpeakerAssignFileUtil.roster_path(project_dir)),
                           Path(SpeakerAssignFileUtil.voices_path(project_dir))]
    for path in configuration_paths:
        path.write_text("configuration kept byte-for-byte", encoding="utf-8")
    texts = ["One.", "Two.", "Three."]
    if mutation == "edit":
        texts[1] = "New text."
    elif mutation == "delete":
        del texts[1]
    changed = make_book([(text, Reason.PHRASE_QUOTE_END, DIALOG) for text in texts])
    plan = AttributionInvalidation.plan(project_dir, changed)
    assert isinstance(plan, AttributionInvalidation)
    assert plan.removed_count == removed
    assert SpeakerAssignFileUtil.load_attributions_file(project_dir) == saved
    assert plan.commit() == ""
    assert SpeakerAssignFileUtil.load_attributions_file(project_dir) == AttributionsFile(tuple(saved.valid_ranges(changed)))
    assert all(path.read_text(encoding="utf-8") == "configuration kept byte-for-byte" for path in configuration_paths)


def test_no_saved_assignments_does_not_create_a_cache(tmp_path):
    plan = AttributionInvalidation.plan(str(tmp_path), make_book([]))
    assert isinstance(plan, AttributionInvalidation)
    assert plan.removed_count == 0
    assert plan.commit() == ""
    assert not Path(SpeakerAssignFileUtil.attributions_path(str(tmp_path))).exists()


def test_invalid_cache_is_reported_without_overwriting_it(tmp_path):
    path = Path(SpeakerAssignFileUtil.attributions_path(str(tmp_path)))
    path.parent.mkdir()
    path.write_text("{invalid", encoding="utf-8")
    assert isinstance(AttributionInvalidation.plan(str(tmp_path), make_book([])), str)
    assert path.read_text(encoding="utf-8") == "{invalid"


def test_cleanup_failure_is_returned(tmp_path, monkeypatch):
    plan = AttributionInvalidation(str(tmp_path), AttributionsFile(), 1)
    monkeypatch.setattr(SpeakerAssignFileUtil, "save_attributions_file", lambda *_: "disk full")
    assert plan.commit() == "disk full"
