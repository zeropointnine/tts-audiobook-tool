import unittest

import pytest

from tts_audiobook_tool.app_types import Book, BookSection, SectionMarkerMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound import m4b_chapter_util
from tts_audiobook_tool.sound.audio_writers import Chapter

# Export code paths: must never launch an FFmpeg executable
pytestmark = pytest.mark.no_ffmpeg_exe


class TestM4bChapterUtil(unittest.TestCase):
    def make_phrase_group(self, text: str) -> PhraseGroup:
        return PhraseGroup([Phrase(text, Reason.SENTENCE)])

    def test_make_chapters_uses_book_section_starts_and_titles(self):
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

        chapters = m4b_chapter_util.make_chapters(
            project=project,
            durations=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        self.assertEqual(chapters, [
            Chapter(0.0, 3.0, "Part One"),
            Chapter(3.0, 6.0, "Part Two"),
            Chapter(6.0, 15.0, "Part Three"),
        ])

    def test_make_chapters_ignores_markers_and_chapter_mode(self):
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

        chapters = m4b_chapter_util.make_chapters(
            project=project,
            durations=[1.0, 1.0, 1.0, 1.0],
        )

        # Book sections win over the FILES-mode markers at 1 and 3
        self.assertEqual(chapters, [
            Chapter(0.0, 2.0, "Book A"),
            Chapter(2.0, 4.0, "Book B"),
        ])

    def test_make_chapters_uses_markers_for_single_section_books(self):
        project = Project.model_validate({
            "book": Book(sections=[
                BookSection(title="Only Section", phrase_groups=[
                    self.make_phrase_group("Chapter One"),
                    self.make_phrase_group("Alpha."),
                    self.make_phrase_group("Chapter Two"),
                    self.make_phrase_group("Beta."),
                ]),
            ]),
            "markers": [2],
            "chapter_mode": SectionMarkerMode.BOOKMARKS.id,
        })

        self.assertTrue(m4b_chapter_util.has_multiple_chapters(project, 0, 3))

        chapters = m4b_chapter_util.make_chapters(
            project=project,
            durations=[1.0, 2.0, 3.0, 4.0],
        )

        self.assertEqual(chapters, [
            Chapter(0.0, 3.0, "Chapter One"),
            Chapter(3.0, 10.0, "Chapter Two"),
        ])

    def test_make_chapters_uses_only_overlapping_marker_chapters_for_output_range(self):
        project = Project.model_validate({
            "book": Book(sections=[
                BookSection(title="Only Section", phrase_groups=[
                    self.make_phrase_group("One"),
                    self.make_phrase_group("Two"),
                    self.make_phrase_group("Three"),
                    self.make_phrase_group("Four"),
                    self.make_phrase_group("Five"),
                ]),
            ]),
            "markers": [2, 4],
            "chapter_mode": SectionMarkerMode.BOOKMARKS.id,
        })

        chapters = m4b_chapter_util.make_chapters(
            project=project,
            durations=[10.0, 20.0, 30.0],
            index_start=2,
            index_end=4,
        )

        self.assertEqual(chapters, [
            Chapter(0.0, 30.0, "Three"),
            Chapter(30.0, 60.0, "Five"),
        ])

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

        chapters = m4b_chapter_util.make_chapters(
            project=project,
            durations=[1.0, 2.0],
        )

        self.assertEqual(chapters, [])


if __name__ == "__main__":
    unittest.main()
