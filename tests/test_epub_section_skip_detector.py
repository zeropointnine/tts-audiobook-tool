import unittest

from tts_audiobook_tool.text_ops.epub_section_skip_detector import EpubSectionSkipDetector


class StubEpubItem:
    def __init__(self, properties=None):
        self.properties = properties or []


def make_toc_links(count=6) -> str:
    links = "".join(f'<li><a href="chapter{i}.xhtml">Chapter {i}</a></li>' for i in range(1, count + 1))
    return f"<html><body><h1>Contents</h1><ol>{links}</ol></body></html>"


class TestEpubSectionSkipDetector(unittest.TestCase):
    def test_detect_publication_metadata_skip_by_href_when_early(self):
        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=0,
            readable_spine_count=10,
            href="Text/copyright.xhtml",
            title="Copyright",
            html="<html><body><p>Copyright page</p></body></html>",
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("publication metadata", decision.reason)

    def test_detect_publication_metadata_skips_labeled_section_in_back_scan_window(self):
        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=16,
            readable_spine_count=20,
            href="Text/copyright.xhtml",
            title="Copyright",
            html="<html><body><p>Copyright page</p></body></html>",
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("href or title", decision.reason)

    def test_detect_publication_metadata_does_not_skip_labeled_section_in_middle(self):
        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=EpubSectionSkipDetector.FRONT_MATTER_SKIP_SCAN_LIMIT,
            readable_spine_count=20,
            href="Text/copyright.xhtml",
            title="Copyright",
            html="<html><body><p>Copyright page</p></body></html>",
        )

        self.assertFalse(decision.should_skip)

    def test_detect_publication_metadata_does_not_use_content_signals_after_scan_limit(self):
        html = """
        <html><body>
            <p>Copyright © 2024 Example Press.</p>
            <p>All rights reserved.</p>
            <p>ISBN 978-1-2345-6789-0</p>
        </body></html>
        """

        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=EpubSectionSkipDetector.FRONT_MATTER_SKIP_SCAN_LIMIT,
            readable_spine_count=20,
            href="Text/backmatter.xhtml",
            title="",
            html=html,
        )

        self.assertFalse(decision.should_skip)
        self.assertEqual(decision.reason, "")

    def test_detect_publication_metadata_skip_by_multiple_content_signals_when_early(self):
        html = """
        <html><body>
            <p>Copyright © 2024 Example Press.</p>
            <p>All rights reserved.</p>
            <p>ISBN 978-1-2345-6789-0</p>
        </body></html>
        """

        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=2,
            readable_spine_count=10,
            href="Text/section01.xhtml",
            title="",
            html=html,
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("text signals", decision.reason)

    def test_detect_publication_metadata_does_not_skip_real_chapter_with_single_copyright_word(self):
        html = """
        <html><body>
            <h1>Chapter 3</h1>
            <p>The lawyer mentioned copyright once during a long conversation.</p>
            <p>Then the story continued without any publication metadata.</p>
        </body></html>
        """

        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=3,
            readable_spine_count=10,
            href="Text/chapter03.xhtml",
            title="Chapter 3",
            html=html,
        )

        self.assertFalse(decision.should_skip)

    def test_detect_publication_metadata_does_not_skip_ordinary_early_chapter(self):
        html = """
        <html><body>
            <h1>Prologue</h1>
            <p>The rain started before dawn.</p>
        </body></html>
        """

        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/prologue.xhtml",
            title="Prologue",
            html=html,
        )

        self.assertFalse(decision.should_skip)

    def test_detect_publication_metadata_does_not_skip_epigraph_section(self):
        html = """
        <html><body>
            <p>First, reality was suspended. All breaches to Inca protocol occurred at once: the rules governing personal contact (visual, oral, and corporal), drinking, and eating were broken.</p>
            <p>—Gonzolo Lamana, in “Beyond Exoticization and Likeness: Alterity and the Production of Sense in a Colonial Encounter,” Comparative Studies in Society and History 47, no. 1 (2005): 4–39</p>
            <p>To ravage, to slaughter, to usurp under false titles—this they name empire; and where they make a desert, they call it peace.</p>
            <p>—Tacitus (quoting Calgacus), Agricola 30</p>
        </body></html>
        """

        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/epigraph.xhtml",
            title="epigraph",
            html=html,
        )

        self.assertFalse(decision.should_skip)

    def test_detect_publication_metadata_skip_imprint_by_title_when_early(self):
        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/section001.xhtml",
            title="Publisher Imprint",
            html="<html><body><p>Example Press</p></body></html>",
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("href or title", decision.reason)

    def test_detect_publication_metadata_skip_colophon_by_href_when_early(self):
        decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/colophon.xhtml",
            title="",
            html="<html><body><p>Publisher credits</p></body></html>",
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("href or title", decision.reason)

    def test_detect_table_of_contents_skip_by_href_and_link_structure(self):
        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/toc.xhtml",
            title="",
            html=make_toc_links(),
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("href/title", decision.reason)

    def test_detect_table_of_contents_skip_by_heading_and_link_structure(self):
        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/section001.xhtml",
            title="",
            html=make_toc_links(),
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("heading", decision.reason)

    def test_detect_table_of_contents_skip_by_link_density_and_short_lines(self):
        html = "<html><body>" + "".join(f'<p><a href="c{i}.xhtml">Part {i}</a></p>' for i in range(1, 7)) + "</body></html>"

        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/frontmatter.xhtml",
            title="",
            html=html,
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("link density", decision.reason)

    def test_detect_table_of_contents_does_not_skip_short_lines_without_links(self):
        html = "<html><body>" + "".join(f"<p>Short line {i}</p>" for i in range(1, 10)) + "</body></html>"

        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/frontmatter.xhtml",
            title="",
            html=html,
        )

        self.assertFalse(decision.should_skip)

    def test_detect_table_of_contents_does_not_skip_chapter_with_few_links(self):
        html = """
        <html><body>
            <h1>Chapter 1</h1>
            <p>See <a href="note.xhtml">note</a> for details.</p>
            <p>The story continued.</p>
        </body></html>
        """

        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=1,
            readable_spine_count=10,
            href="Text/chapter01.xhtml",
            title="Chapter 1",
            html=html,
        )

        self.assertFalse(decision.should_skip)

    def test_detect_table_of_contents_skips_labeled_toc_in_back_scan_window(self):
        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=16,
            readable_spine_count=20,
            href="Text/toc.xhtml",
            title="Contents",
            html=make_toc_links(),
        )

        self.assertTrue(decision.should_skip)
        self.assertIn("table of contents", decision.reason)

    def test_detect_table_of_contents_does_not_skip_labeled_toc_in_middle(self):
        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=EpubSectionSkipDetector.FRONT_MATTER_SKIP_SCAN_LIMIT,
            readable_spine_count=20,
            href="Text/toc.xhtml",
            title="Contents",
            html=make_toc_links(),
        )

        self.assertFalse(decision.should_skip)

    def test_detect_table_of_contents_does_not_use_weak_link_density_after_scan_limit(self):
        html = "<html><body>" + "".join(f'<p><a href="c{i}.xhtml">Part {i}</a></p>' for i in range(1, 7)) + "</body></html>"

        decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
            readable_spine_index=EpubSectionSkipDetector.FRONT_MATTER_SKIP_SCAN_LIMIT,
            readable_spine_count=20,
            href="Text/backmatter.xhtml",
            title="",
            html=html,
        )

        self.assertFalse(decision.should_skip)

    def test_is_navigation_document_by_item_id(self):
        self.assertTrue(EpubSectionSkipDetector.is_navigation_document(
            item_id="nav",
            href="Text/chapter01.xhtml",
            item=StubEpubItem(),
        ))

    def test_is_navigation_document_by_basename(self):
        self.assertTrue(EpubSectionSkipDetector.is_navigation_document(
            item_id="item1",
            href="OPS/nav.xhtml",
            item=StubEpubItem(),
        ))

    def test_is_navigation_document_by_property(self):
        self.assertTrue(EpubSectionSkipDetector.is_navigation_document(
            item_id="item1",
            href="OPS/document.xhtml",
            item=StubEpubItem(properties=["nav"]),
        ))

    def test_is_likely_empty_non_reading_section(self):
        self.assertTrue(EpubSectionSkipDetector.is_likely_empty_non_reading_section(
            href="Text/title-page.xhtml",
            title="Title Page",
        ))

    def test_does_not_identify_ordinary_section_as_likely_empty_non_reading(self):
        self.assertFalse(EpubSectionSkipDetector.is_likely_empty_non_reading_section(
            href="Text/chapter01.xhtml",
            title="Chapter 1",
        ))


if __name__ == "__main__":
    unittest.main()
