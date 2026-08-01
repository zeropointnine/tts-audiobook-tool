from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.text_edit_session import (
    PhraseGroupSplitPoint,
    TextEditSession,
)


def make_phrase_group(*texts: str, voice_index: int = -1) -> PhraseGroup:
    return PhraseGroup(
        [Phrase(text, Reason.SENTENCE) for text in texts],
        voice_index=voice_index,
    )


def make_book() -> Book:
    return Book(
        sections=[
            BookSection(
                title="One",
                phrase_groups=[make_phrase_group("A."), make_phrase_group("B.")],
            ),
            BookSection(
                title="Two",
                phrase_groups=[make_phrase_group("C."), make_phrase_group("D.")],
            ),
        ],
        title="Book title",
        text_source_kind="epub",
        audio_source_kind="generated",
    )


def test_session_is_detached_and_preserves_book_metadata() -> None:
    book = make_book()
    session = TextEditSession(book)

    session.sections[0].title = "Changed"
    session.phrase_groups[0].phrase_group.phrases[0].text = "Changed."

    assert book.sections[0].title == "One"
    assert book.phrase_groups[0].text == "A."
    staged_book = session.to_book()
    assert staged_book.title == book.title
    assert staged_book.text_source_kind == book.text_source_kind
    assert staged_book.audio_source_kind == book.audio_source_kind
    assert session.has_changes is True


def test_delete_ignores_unknown_ids_and_prunes_empty_sections() -> None:
    session = TextEditSession(make_book())
    ids = {item.item_id for item in session.sections[0].phrase_groups}

    result = session.delete_phrase_groups(ids | {999_999})

    assert result.changed is True
    assert result.earliest_affected_original_index == 0
    assert result.deleted_count == 2
    assert [section.title for section in session.sections] == ["Two"]
    assert [item.phrase_group.text for item in session.phrase_groups] == ["C.", "D."]


def test_delete_preserves_an_unrelated_preexisting_empty_section() -> None:
    book = make_book()
    book.sections.append(BookSection(title="Already empty", phrase_groups=[]))
    session = TextEditSession(book)

    result = session.delete_phrase_groups({session.phrase_groups[0].item_id})

    assert result.changed is True
    assert [section.title for section in session.sections] == [
        "One",
        "Two",
        "Already empty",
    ]


def test_delete_only_phrase_group_deletes_it_and_its_section() -> None:
    session = TextEditSession(
        Book(sections=[BookSection(phrase_groups=[make_phrase_group("Only.")])])
    )
    only_item = session.phrase_groups[0]

    result = session.delete_phrase_groups({only_item.item_id})

    assert result.changed is True
    assert result.deleted_count == 1
    assert result.focus_item_id is None
    assert session.sections == []
    assert session.phrase_groups == []
    assert session.has_changes is True


def test_delete_all_phrase_groups_deletes_all_owning_sections() -> None:
    session = TextEditSession(make_book())

    result = session.delete_phrase_groups(
        {phrase_group.item_id for phrase_group in session.phrase_groups}
    )

    assert result.changed is True
    assert result.earliest_affected_original_index == 0
    assert result.deleted_count == 4
    assert result.focus_item_id is None
    assert session.sections == []
    assert session.phrase_groups == []


def test_delete_section_deletes_its_phrase_groups_and_the_section() -> None:
    session = TextEditSession(make_book())
    section_item_id = session.sections[0].item_id

    result = session.delete_section(section_item_id)

    assert result.changed is True
    assert result.deleted_count == 2
    assert result.earliest_affected_original_index == 0
    assert [section.title for section in session.sections] == ["Two"]
    assert [item.phrase_group.text for item in session.phrase_groups] == ["C.", "D."]


def test_delete_section_can_delete_all_remaining_phrase_groups() -> None:
    session = TextEditSession(
        Book(
            sections=[
                BookSection(
                    title="Only text",
                    phrase_groups=[make_phrase_group("A."), make_phrase_group("B.")],
                ),
                BookSection(title="Empty", phrase_groups=[]),
            ]
        )
    )

    result = session.delete_section(session.sections[0].item_id)

    assert result.changed is True
    assert result.deleted_count == 2
    assert result.focus_item_id is None
    assert [section.title for section in session.sections] == ["Empty"]
    assert session.phrase_groups == []


def test_delete_empty_section_removes_it_without_a_deleted_line_count() -> None:
    session = TextEditSession(
        Book(
            sections=[
                BookSection(title="Text", phrase_groups=[make_phrase_group("A.")]),
                BookSection(title="Empty", phrase_groups=[]),
            ]
        )
    )

    result = session.delete_section(session.sections[1].item_id)

    assert result.changed is True
    assert result.deleted_count == 0
    assert [section.title for section in session.sections] == ["Text"]
    assert result.focus_item_id == session.phrase_groups[0].item_id


def test_split_at_phrase_boundary_preserves_voice_reason_and_lineage() -> None:
    source = make_phrase_group("First. ", "Second.", voice_index=3)
    session = TextEditSession(
        Book(sections=[BookSection(title="One", phrase_groups=[source])])
    )
    source_item = session.phrase_groups[0]

    result = session.split_phrase_group(
        source_item.item_id,
        PhraseGroupSplitPoint(phrase_boundary=1),
    )

    assert result.changed is True
    assert result.earliest_affected_original_index == 0
    assert [item.phrase_group.text for item in session.phrase_groups] == [
        "First. ",
        "Second.",
    ]
    assert [item.phrase_group.voice_index for item in session.phrase_groups] == [3, 3]
    assert [item.original_index for item in session.phrase_groups] == [0, 0]
    assert session.phrase_groups[0].phrase_group.last_reason == Reason.SENTENCE
    assert session.phrase_groups[1].phrase_group.last_reason == Reason.SENTENCE
    assert result.focus_item_id == session.phrase_groups[1].item_id


def test_split_rejects_endpoints_and_tracks_earliest_mutation() -> None:
    session = TextEditSession(
        Book(
            sections=[
                BookSection(
                    phrase_groups=[
                        make_phrase_group("A."),
                        make_phrase_group("B. ", "C."),
                        make_phrase_group("D. ", "E."),
                    ]
                )
            ]
        )
    )
    second, third = session.phrase_groups[1:]

    invalid = session.split_phrase_group(
        second.item_id,
        PhraseGroupSplitPoint(phrase_boundary=0),
    )
    assert invalid.changed is False

    session.split_phrase_group(third.item_id, PhraseGroupSplitPoint(1))
    session.split_phrase_group(second.item_id, PhraseGroupSplitPoint(1))

    assert session.earliest_affected_original_index == 1
