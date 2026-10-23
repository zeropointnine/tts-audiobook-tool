import dataclasses

import pytest

from tts_audiobook_tool.speaker_assign.roster import (
    NarrativePerson,
    Prominence,
    Roster,
    RosterCharacter,
    RosterNarration,
)


# A roster in the on-disk shape the prototype's speaker_attribution.json uses.
STORED_ROSTER = {
    "narration": {"person": "first", "notes": "A physician arrives at a remote château."},
    "characters": [
        {
            "id": "NARRATOR",
            "name": "The Physician",
            "aliases": ["doctor", "Simone"],
            "gender": "unknown",
            "description": "The first-person narrator.",
            "is_narrator": True,
            "prominence": "major",
        },
        {
            "id": "OLD_DOC",
            "name": "Old Doctor",
            "aliases": ["Old Doc"],
            "gender": "male",
            "description": "The previous physician.",
            "is_narrator": False,
            "prominence": "minor",
        },
    ],
}


def test_round_trip_preserves_stored_form() -> None:
    # Loading and re-serializing a stored roster gives back the same JSON shape,
    # so existing prototype cache files keep working.
    roster = Roster.from_dict(STORED_ROSTER)
    assert roster.to_dict() == STORED_ROSTER
    assert roster.narration == RosterNarration(NarrativePerson.FIRST, STORED_ROSTER["narration"]["notes"])
    assert roster.characters[0].prominence is Prominence.MAJOR


def test_lookups() -> None:
    roster = Roster.from_dict(STORED_ROSTER)
    assert roster.ids == {"NARRATOR", "OLD_DOC"}
    assert roster.narrator is not None and roster.narrator.id == "NARRATOR"
    assert roster.get("OLD_DOC") is roster.characters[1]
    assert roster.get("NOBODY") is None


@pytest.mark.parametrize("bad_id", ["Old Doc", "OLD DOC", "old_doc", "OLD__DOC", "_OLD", "OLD_", "", "OLD-DOC"])
def test_invalid_ids_are_rejected(bad_id: str) -> None:
    # Ids are canonical speaker labels: uppercase words joined by single
    # underscores, never spaces or other punctuation.
    with pytest.raises(ValueError):
        RosterCharacter(id=bad_id, name="x")


@pytest.mark.parametrize("reserved", ["OTHER", "UNKNOWN", "NOT_SPEECH"])
def test_reserved_answers_cannot_be_ids(reserved: str) -> None:
    # The reserved attribution answers must stay distinguishable from characters.
    with pytest.raises(ValueError):
        RosterCharacter(id=reserved, name="x")


def test_invalid_id_in_stored_data_fails_to_load() -> None:
    # A hand-edited cache with a spaced id is rejected rather than loaded.
    stored = {"characters": [{"id": "OLD DOC", "name": "Old Doctor"}]}
    with pytest.raises(ValueError):
        Roster.from_dict(stored)


def test_duplicate_ids_and_second_narrator_are_rejected() -> None:
    first = RosterCharacter(id="BARON", name="Baron", is_narrator=True)
    with pytest.raises(ValueError, match="Duplicate"):
        Roster((first, RosterCharacter(id="BARON", name="Other baron")))
    with pytest.raises(ValueError, match="narrator"):
        Roster((first, RosterCharacter(id="DIDIER", name="Didier", is_narrator=True)))


def test_optional_fields_load_tolerantly() -> None:
    # Missing or malformed optional values fall back to defaults instead of
    # failing: unknown tier, non-list aliases, non-string alias entries,
    # non-boolean narrator flag, missing narration.
    roster = Roster.from_dict({"characters": [
        {"id": "BARON", "prominence": "legendary", "aliases": "the baron", "is_narrator": "yes"},
        {"id": "DIDIER", "name": "Didier", "aliases": ["Did", 7]},
    ]})
    baron, didier = roster.characters
    assert baron.name == "BARON" and baron.prominence is Prominence.UNKNOWN
    assert baron.aliases == () and baron.is_narrator is False
    assert didier.aliases == ("Did",)
    assert roster.narration == RosterNarration()


def test_instances_are_immutable_and_hashable() -> None:
    # A list passed for aliases is stored as a tuple, so characters stay hashable;
    # edits go through dataclasses.replace.
    character = RosterCharacter(id="BARON", name="Baron", aliases=["the baron"])  # type: ignore[arg-type]
    assert character.aliases == ("the baron",)
    assert hash(character) == hash(dataclasses.replace(character))
    with pytest.raises(dataclasses.FrozenInstanceError):
        character.name = "x"  # type: ignore[misc]


def test_name_keys_normalize_for_matching() -> None:
    character = RosterCharacter(id="OLD_DOC", name="Old  Doctor", aliases=("OLD DOC",))
    assert character.name_keys() == {"old doctor", "old doc"}


def test_prominence_rank_orders_tiers() -> None:
    ranks = [p.rank for p in (Prominence.MAJOR, Prominence.SUPPORTING, Prominence.MINOR, Prominence.UNKNOWN)]
    assert ranks == sorted(ranks, reverse=True)
