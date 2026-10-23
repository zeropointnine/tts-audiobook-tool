import pytest

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.roster import NarrativePerson, Prominence
from tts_audiobook_tool.speaker_assign.roster_util import RosterUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book


def make_test_book():
    return make_book(
        [
            ("He said, ", Reason.PHRASE, NARRATION),
            ("“Hello.”\n\n", Reason.PARAGRAPH, DIALOG),
        ],
        [
            ("“Bye.”\n\n", Reason.PARAGRAPH, DIALOG),
        ],
        titles=["One", "Two"],
    )


def test_build_message_heads_sections_and_tags_utterances() -> None:
    message = RosterUtil.build_message(make_test_book())
    assert "===== Section 1: One =====\nHe said, <q>“Hello.”</q>" in message
    assert "===== Section 2: Two =====\n<q>“Bye.”</q>" in message
    assert message.endswith("Now respond with the JSON roster.")


def test_normalize_cleans_ids_and_parses_fields() -> None:
    # Ids are uppercased with punctuation/space runs collapsed to one
    # underscore; tolerant fields fall back to defaults.
    normalized = RosterUtil.normalize_roster({
        "narration": {"person": "First", "notes": "A diary."},
        "characters": [
            {"id": "old  doc!", "name": "The old doctor", "aliases": ["doc"], "prominence": "MAJOR"},
            {"id": "maid", "name": "Maid", "prominence": "bogus"},
        ],
    })
    roster = normalized.roster
    assert [c.id for c in roster.characters] == ["OLD_DOC", "MAID"]
    assert roster.characters[0].prominence == Prominence.MAJOR
    assert roster.characters[1].prominence == Prominence.UNKNOWN
    assert roster.narration.person == NarrativePerson.FIRST
    assert normalized.notes == ()


def test_normalize_drops_bad_ids_and_resolves_duplicates() -> None:
    # A reserved or empty id is dropped; a duplicate id for the same character
    # (its name among the first's aliases) is merged, while one for a
    # different character is renamed. Each adjustment leaves a note.
    normalized = RosterUtil.normalize_roster({
        "characters": [
            {"id": "GUARD", "name": "Gate guard", "aliases": ["the sentry"], "gender": "unknown"},
            {"id": "UNKNOWN", "name": "Someone"},
            {"id": "!!", "name": "Nobody"},
            {"id": "guard", "name": "the sentry", "gender": "male", "prominence": "major"},
            {"id": "GUARD", "name": "Mine guard"},
        ],
    })
    characters = normalized.roster.characters
    assert [c.id for c in characters] == ["GUARD", "GUARD_2"]
    assert characters[0].gender == "male"
    assert characters[0].prominence == Prominence.MAJOR
    assert characters[1].name == "Mine guard"
    assert len(normalized.notes) == 4


def test_normalize_keeps_only_the_first_narrator() -> None:
    normalized = RosterUtil.normalize_roster({
        "characters": [
            {"id": "NARRATOR", "name": "I", "is_narrator": True},
            {"id": "OTHER_ME", "name": "Me", "is_narrator": True},
        ],
    })
    assert normalized.roster.narrator is not None
    assert normalized.roster.narrator.id == "NARRATOR"


@pytest.mark.parametrize("value", [
    {},
    {"characters": []},
    {"characters": [{"id": "OTHER", "name": "x"}, "not an object"]},
])
def test_normalize_rejects_a_reply_without_usable_characters(value) -> None:
    with pytest.raises(ValueError):
        RosterUtil.normalize_roster(value)


def test_is_stale_compares_section_text_hashes() -> None:
    book = make_test_book()
    hashes = UtteranceUtil.section_text_hashes(book)
    assert not RosterUtil.is_stale(hashes, book)
    assert RosterUtil.is_stale(hashes[:1], book)
    assert RosterUtil.is_stale(("x", hashes[1]), book)
