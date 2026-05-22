import unittest

from tts_audiobook_tool.app_types import Book, BookSection, SectionMarkerMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound import m4b_chapter_util


class TestM4bChapterUtil(unittest.TestCase):
    def make_phrase_group(self, text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])

    def test_make_metadata_uses_book_section_starts_and_titles(self):
        project = Project.model_validate({
            "book": Book(sections=[
                BookSection(title="Part One", phrase_groups=[
                    self.make_phrase_group("One."),
                    self.make_phrase_group("Two."),
                ]),
                BookSection(title="Part Two", phrase_groups=[
                    self.make_phrase_group("Three."),
                ]),
                BookSection(title="Part Three", phrase_groups=[
                    self.make_phrase_group("Four."),
                    self.make_phrase_group("Five."),
                ]),
            ]),
        })

        metadata = m4b_chapter_util.make_metadata(
            project=project,
            durations=[1.0, 2.0, 3.0, 4.0, 5.0],
            file_title="Test Book",
        )

        self.assertEqual(metadata.count("[CHAPTER]"), 3)
        self.assertIn("START=0\nEND=3000\ntitle=Part One", metadata)
        self.assertIn("START=3000\nEND=6000\ntitle=Part Two", metadata)
        self.assertIn("START=6000\nEND=15000\ntitle=Part Three", metadata)

    def test_make_metadata_ignores_markers_and_chapter_mode(self):
        project = Project.model_validate({
            "book": Book(sections=[
                BookSection(title="Book A", phrase_groups=[
                    self.make_phrase_group("A1."),
                    self.make_phrase_group("A2."),
                ]),
                BookSection(title="Book B", phrase_groups=[
                    self.make_phrase_group("B1."),
                    self.make_phrase_group("B2."),
                ]),
            ]),
            "markers": [1, 3],
            "chapter_mode": SectionMarkerMode.FILES.id,
        })

        metadata = m4b_chapter_util.make_metadata(
            project=project,
            durations=[1.0, 1.0, 1.0, 1.0],
            file_title="Test Book",
        )

        self.assertEqual(metadata.count("[CHAPTER]"), 2)
        self.assertIn("START=0\nEND=2000\ntitle=Book A", metadata)
        self.assertIn("START=2000\nEND=4000\ntitle=Book B", metadata)
        self.assertNotIn("START=1000", metadata)
        self.assertNotIn("START=3000", metadata)
        self.assertNotIn("title=A2.", metadata)
        self.assertNotIn("title=B2.", metadata)

    def test_single_book_section_does_not_create_m4b_chapters(self):
        project = Project.model_validate({
            "book": Book(sections=[
                BookSection(title="Only Section", phrase_groups=[
                    self.make_phrase_group("One."),
                    self.make_phrase_group("Two."),
                ]),
            ]),
        })

        self.assertFalse(m4b_chapter_util.has_multiple_chapters(project, 0, 1))

        metadata = m4b_chapter_util.make_metadata(
            project=project,
            durations=[1.0, 2.0],
            file_title="Test Book",
        )

        self.assertNotIn("[CHAPTER]", metadata)


if __name__ == "__main__":
    unittest.main()