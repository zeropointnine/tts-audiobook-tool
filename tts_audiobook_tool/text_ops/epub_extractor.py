from __future__ import annotations

import os
import re
import shutil
import importlib
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Protocol

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.constants import PROJECT_TEXT_EPUB_FILE_NAME
from tts_audiobook_tool.text_ops.epub_section_skip_detector import EpubSectionSkipDetector
from tts_audiobook_tool.l import L
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


@dataclass
class EpubSourceChapter:
    title: str
    href: str
    media_type: str
    html: str


@dataclass
class EpubTextChapter:
    title: str
    href: str
    text: str


@dataclass
class EpubTextExtractionResult:
    text: str
    warnings: list[str] = field(default_factory=list)
    significant_warnings: list[str] = field(default_factory=list)


@dataclass
class EpubTextExtractionStats:
    inline_whitespace_repairs: int = 0


@dataclass
class EpubImportResult:
    phrase_groups: list[PhraseGroup]
    raw_text: str
    section_dividers: list[int]
    chapters: list[EpubTextChapter]
    book_title: str = ""
    warnings: list[str] = field(default_factory=list)
    significant_warnings: list[str] = field(default_factory=list)


class EpubChapterTextExtractor(Protocol):
    def extract_text(self, chapter: EpubSourceChapter) -> EpubTextExtractionResult:
        ...


class BeautifulSoupEpubChapterTextExtractor:
    SKIP_TAGS = {
        "audio",
        "button",
        "canvas",
        "form",
        "head",
        "iframe",
        "img",
        "input",
        "nav",
        "noscript",
        "object",
        "script",
        "select",
        "style",
        "svg",
        "textarea",
        "title",
        "video",
    }
    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    INLINE_WHITESPACE_REPAIR_WARNING_PREFIX = "EPUB text extraction repaired inline spacing"
    INLINE_WHITESPACE_REPAIR_WARNING_THRESHOLD = 10
    INLINE_TAGS = {
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "cite",
        "code",
        "data",
        "dfn",
        "em",
        "i",
        "kbd",
        "mark",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
        "wbr",
    }

    def extract_text(self, chapter: EpubSourceChapter) -> EpubTextExtractionResult:
        BeautifulSoup = self.import_beautiful_soup()
        soup = BeautifulSoup(chapter.html, "html.parser")
        stats = EpubTextExtractionStats()
        warnings: list[str] = []
        significant_warnings: list[str] = []

        for tag in soup.find_all(list(self.SKIP_TAGS)):
            tag.decompose()

        root = soup.body or soup
        text = self.node_to_text(root, stats)
        text = self.normalize_output_text(text)

        if stats.inline_whitespace_repairs >= self.INLINE_WHITESPACE_REPAIR_WARNING_THRESHOLD:
            warning = (
                f"{self.INLINE_WHITESPACE_REPAIR_WARNING_PREFIX} {stats.inline_whitespace_repairs} times "
                f"in {chapter.href}. Some inline spacing in this EPUB needed cleanup during import; "
                "review imported text if spacing looks unusual."
            )
            warnings.append(warning)
            significant_warnings.append(warning)

        if not text:
            warning = f"No readable body text found in {chapter.href}"
            warnings.append(warning)
            if not self.is_likely_non_reading_chapter(chapter):
                significant_warnings.append(warning)

        return EpubTextExtractionResult(
            text=text,
            warnings=warnings,
            significant_warnings=significant_warnings,
        )

    @classmethod
    def is_likely_non_reading_chapter(cls, chapter: EpubSourceChapter) -> bool:
        if EpubSectionSkipDetector.is_likely_empty_non_reading_section(chapter.href, chapter.title):
            return True
        return cls.is_image_only_chapter(chapter.html)

    @classmethod
    def is_image_only_chapter(cls, html: str) -> bool:
        BeautifulSoup = cls.import_beautiful_soup()
        soup = BeautifulSoup(html, "html.parser")
        root = soup.body or soup
        if not root.find("img"):
            return False

        for tag in root.find_all(list(cls.SKIP_TAGS)):
            tag.decompose()

        text = cls.normalize_output_text(root.get_text(" "))
        return not text

    @staticmethod
    def import_beautiful_soup() -> Any:
        try:
            module = importlib.import_module("bs4")
            return getattr(module, "BeautifulSoup")
        except Exception as e:
            raise ImportError("Missing dependency beautifulsoup4. Reinstall requirements for EPUB import support.") from e

    def node_to_text(self, node: Any, stats: EpubTextExtractionStats | None = None) -> str:
        stats = stats or EpubTextExtractionStats()
        pieces: list[str] = []
        self.append_node_text(node, pieces, stats)
        return "".join(pieces)

    def append_node_text(self, node: Any, pieces: list[str], stats: EpubTextExtractionStats) -> None:
        name = getattr(node, "name", None)

        if name in self.SKIP_TAGS:
            return
        if name == "br":
            pieces.append("\n")
            return
        if name == "hr":
            pieces.append("\n\n* * *\n\n")
            return
        if name == "li":
            pieces.append("\n- ")

        is_block = name in self.BLOCK_TAGS
        if is_block and name != "li":
            pieces.append("\n\n")

        if not hasattr(node, "children"):
            self.append_text_node(str(node), pieces)
        else:
            for child in node.children:
                child_name = getattr(child, "name", None)
                if child_name is None:
                    self.append_text_node(str(child), pieces, name, child, stats)
                else:
                    self.append_node_text(child, pieces, stats)

        if is_block:
            pieces.append("\n\n")

    @classmethod
    def append_text_node(
            cls,
            text: str,
            pieces: list[str],
            parent_name: str | None = None,
            text_node: Any | None = None,
            stats: EpubTextExtractionStats | None = None,
    ) -> None:
        if not text.strip() and "\n" in text:
            if cls.should_repair_inline_whitespace_separator(parent_name, pieces, text_node):
                # Some EPUB conversion pipelines produce very ugly but technically recoverable markup where
                # normal inline word/sentence separators are represented as whitespace-only inline elements,
                # for example:
                #
                #   <span>Will it happen?</span><span>\n</span><span>That is the idea.</span>
                #
                # Treating all such newline-only text as structural indentation concatenates neighboring
                # sentences (`happen?That`). Treating all HTML newlines as paragraph breaks would be worse,
                # because most EPUB XHTML is pretty-printed with irrelevant indentation between block tags.
                # This branch is intentionally narrow: only whitespace-only text inside known inline tags,
                # with readable text already emitted and readable sibling content still ahead, is collapsed to
                # one ordinary space. The counter feeds a thresholded user-facing warning so highly affected
                # EPUBs remain visible instead of being silently "fixed".
                if pieces and not pieces[-1].endswith((" ", "\n")):
                    pieces.append(" ")
                if stats is not None:
                    stats.inline_whitespace_repairs += 1
            return
        pieces.append(text)

    @classmethod
    def should_repair_inline_whitespace_separator(
            cls,
            parent_name: str | None,
            pieces: list[str],
            text_node: Any | None,
    ) -> bool:
        if parent_name not in cls.INLINE_TAGS:
            return False
        if not pieces or not "".join(pieces).strip():
            return False
        return cls.has_readable_following_sibling(text_node)

    @classmethod
    def has_readable_following_sibling(cls, text_node: Any | None) -> bool:
        sibling = cls.get_next_sibling_after_inline_whitespace_node(text_node)
        while sibling is not None:
            name = getattr(sibling, "name", None)
            if name in cls.SKIP_TAGS:
                sibling = getattr(sibling, "next_sibling", None)
                continue
            if name is None:
                if str(sibling).strip():
                    return True
            elif cls.normalize_inline_text(sibling.get_text(" ")):
                return True
            sibling = getattr(sibling, "next_sibling", None)
        return False

    @classmethod
    def get_next_sibling_after_inline_whitespace_node(cls, text_node: Any | None) -> Any | None:
        sibling = getattr(text_node, "next_sibling", None)
        if sibling is not None:
            return sibling

        parent = getattr(text_node, "parent", None)
        while parent is not None and getattr(parent, "name", None) in cls.INLINE_TAGS:
            sibling = getattr(parent, "next_sibling", None)
            if sibling is not None:
                return sibling
            parent = getattr(parent, "parent", None)
        return None

    @staticmethod
    def normalize_inline_text(text: str) -> str:
        text = unescape(text).replace("\xa0", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def normalize_output_text(text: str) -> str:
        text = unescape(text).replace("\xa0", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(line)
        text = "\n".join(lines)
        text = re.sub(
            r"\n{3,}",
            lambda match: "\n\n\n" if len(match.group(0)) >= 5 else "\n\n",
            text,
        )
        return text.strip()


class EpubExtractor:
    DEFAULT_EXTRACTOR = BeautifulSoupEpubChapterTextExtractor()

    @staticmethod
    def import_epub(
            epub_path: str,
            max_words: int,
            segmentation_strategy: SegmentationStrategy,
            language_code: str,
            extractor: EpubChapterTextExtractor | None = None
    ) -> EpubImportResult:
        source_chapters, book_title, warnings, significant_warnings = EpubExtractor.load_source_chapters(epub_path)
        EpubExtractor.log_warnings(warnings)
        EpubExtractor.log_warnings([warning for warning in significant_warnings if warning not in warnings])
        extractor = extractor or EpubExtractor.DEFAULT_EXTRACTOR

        text_chapters: list[EpubTextChapter] = []
        did_report_inline_whitespace_repair_warning = False
        for source_chapter in source_chapters:
            result = extractor.extract_text(source_chapter)
            result_warnings, result_significant_warnings, did_report_inline_whitespace_repair_warning = (
                EpubExtractor.filter_repeated_inline_whitespace_repair_warnings(
                    result.warnings,
                    result.significant_warnings,
                    did_report_inline_whitespace_repair_warning,
                )
            )
            EpubExtractor.log_warnings(result_warnings)
            warnings.extend(result_warnings)
            significant_warnings.extend(result_significant_warnings)
            if not result.text:
                continue
            text_chapters.append(EpubTextChapter(
                title=source_chapter.title,
                href=source_chapter.href,
                text=result.text,
            ))

        if EpubExtractor.prepend_book_title_chapter_if_needed(text_chapters, book_title):
            warning = f"Inserted EPUB metadata title at start of imported text: {book_title}"
            EpubExtractor.log_warnings([warning])
            warnings.append(warning)

        phrase_groups: list[PhraseGroup] = []
        section_dividers: list[int] = []
        raw_text_parts: list[str] = []
        has_seen_spine_chapter = False

        for chapter in text_chapters:
            chapter_text = chapter.text.strip()
            if not chapter_text:
                continue
            is_injected_book_title = EpubExtractor.is_injected_book_title_chapter(chapter)
            if not is_injected_book_title:
                if has_seen_spine_chapter:
                    section_dividers.append(len(phrase_groups))
                has_seen_spine_chapter = True
            else:
                # The metadata title can be injected as a pseudo-chapter for text readability,
                # but it is not an EPUB spine entry and should not create a bookmark/file divider
                # before any later real spine entry.
                pass
            chapter_phrase_groups = PhraseGrouper.text_to_groups(
                chapter_text,
                max_words=max_words,
                strategy=segmentation_strategy,
                pysbd_lang=language_code,
            )
            EpubExtractor.mark_last_phrase_as_section(chapter_phrase_groups)
            phrase_groups.extend(chapter_phrase_groups)
            raw_text_parts.append(chapter_text)

        raw_text = "\n\n".join(raw_text_parts)

        if not phrase_groups:
            warning = "EPUB import produced no text segments."
            warnings.append(warning)
            significant_warnings.append(warning)
            L.w(warning)

        return EpubImportResult(
            phrase_groups=phrase_groups,
            raw_text=raw_text,
            section_dividers=section_dividers,
            chapters=text_chapters,
            book_title=book_title,
            warnings=warnings,
            significant_warnings=significant_warnings,
        )

    @staticmethod
    def filter_repeated_inline_whitespace_repair_warnings(
            warnings: list[str],
            significant_warnings: list[str],
            did_report_warning: bool,
    ) -> tuple[list[str], list[str], bool]:
        filtered_warnings: list[str] = []
        filtered_significant_warnings: list[str] = []

        for warning in warnings:
            if EpubExtractor.is_inline_whitespace_repair_warning(warning):
                if did_report_warning:
                    continue
                did_report_warning = True
            filtered_warnings.append(warning)

        for warning in significant_warnings:
            if EpubExtractor.is_inline_whitespace_repair_warning(warning) and warning not in filtered_warnings:
                if did_report_warning:
                    continue
                did_report_warning = True
            filtered_significant_warnings.append(warning)

        return filtered_warnings, filtered_significant_warnings, did_report_warning

    @staticmethod
    def is_inline_whitespace_repair_warning(warning: str) -> bool:
        return warning.startswith(BeautifulSoupEpubChapterTextExtractor.INLINE_WHITESPACE_REPAIR_WARNING_PREFIX)

    @staticmethod
    def prepend_book_title_chapter_if_needed(text_chapters: list[EpubTextChapter], book_title: str) -> bool:
        book_title = BeautifulSoupEpubChapterTextExtractor.normalize_inline_text(book_title)
        if not book_title:
            return False
        if text_chapters and EpubExtractor.text_starts_with_title(text_chapters[0].text, book_title):
            return False
        text_chapters.insert(0, EpubTextChapter(
            title=book_title,
            href="__epub_book_title__",
            text=book_title,
        ))
        return True

    @staticmethod
    def is_injected_book_title_chapter(chapter: EpubTextChapter) -> bool:
        return chapter.href == "__epub_book_title__"

    @staticmethod
    def text_starts_with_title(text: str, book_title: str) -> bool:
        normalized_title = EpubExtractor.normalize_title_match_text(book_title)
        if not normalized_title:
            return False

        prefix = " ".join(text.splitlines()[:4])
        normalized_prefix = EpubExtractor.normalize_title_match_text(prefix)
        return normalized_prefix.startswith(normalized_title)

    @staticmethod
    def normalize_title_match_text(text: str) -> str:
        text = BeautifulSoupEpubChapterTextExtractor.normalize_inline_text(text).lower()
        return re.sub(r"[^a-z0-9]+", "", text)

    @staticmethod
    def mark_last_phrase_as_section(phrase_groups: list[PhraseGroup]) -> None:
        if not phrase_groups:
            return
        last_group = phrase_groups[-1]
        if not last_group.phrases:
            return
        last_phrase = last_group.phrases[-1]
        last_phrase.reason = Reason.SECTION
        # Also add two blank lines at end, matching expectations and behavior for flat text flow
        last_phrase.text = last_phrase.text.rstrip() + "\n\n\n"

    @staticmethod
    def load_source_chapters(epub_path: str) -> tuple[list[EpubSourceChapter], str, list[str], list[str]]:
        if not os.path.exists(epub_path):
            return [], "", [], [f"EPUB file does not exist: {epub_path}"]
        if os.path.splitext(epub_path)[1].lower() != ".epub":
            return [], "", [], [f"File does not have .epub suffix: {epub_path}"]

        try:
            epub = importlib.import_module("ebooklib.epub")
        except Exception as e:
            raise ImportError("Missing dependency EbookLib. Reinstall requirements for EPUB import support.") from e

        try:
            book = epub.read_epub(epub_path)
        except Exception as e:
            message = f"Error reading EPUB: {e}"
            L.e(message)
            return [], "", [], [message]

        book_title = EpubExtractor.extract_book_title(book)
        warnings: list[str] = []
        significant_warnings: list[str] = []
        source_chapter_candidates: list[EpubSourceChapter] = []
        source_chapters: list[EpubSourceChapter] = []

        for spine_item in book.spine:
            item_id = spine_item[0] if isinstance(spine_item, tuple) else spine_item
            item = book.get_item_with_id(item_id)
            if item is None:
                warning = f"EPUB spine item not found: {item_id}"
                warnings.append(warning)
                significant_warnings.append(warning)
                continue

            href = getattr(item, "file_name", "") or str(item_id)
            media_type = getattr(item, "media_type", "")
            html = EpubExtractor.decode_item_content(item)
            if not html:
                warning = f"EPUB spine item has no decodable content: {item_id}"
                warnings.append(warning)
                continue

            title = EpubExtractor.extract_title(html, href)
            if EpubSectionSkipDetector.is_navigation_document(str(item_id), href, item):
                toc_skip_decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
                    readable_spine_index=len(source_chapter_candidates),
                    readable_spine_count=max(1, len(source_chapter_candidates) + 1),
                    href=href,
                    title=title,
                    html=html,
                )
                if toc_skip_decision.should_skip:
                    EpubExtractor.append_skipped_section_warning(
                        warnings,
                        significant_warnings,
                        EpubSourceChapter(
                            title=title,
                            href=href,
                            media_type=media_type,
                            html=html,
                        ),
                        toc_skip_decision.reason,
                    )
                else:
                    warning = f"Skipped EPUB navigation document: {href}"
                    warnings.append(warning)
                continue

            source_chapter_candidates.append(EpubSourceChapter(
                title=title,
                href=href,
                media_type=media_type,
                html=html,
            ))

        readable_spine_count = len(source_chapter_candidates)
        for readable_spine_index, source_chapter in enumerate(source_chapter_candidates):
            publication_metadata_skip_decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
                readable_spine_index,
                readable_spine_count,
                source_chapter.href,
                source_chapter.title,
                source_chapter.html,
            )
            if publication_metadata_skip_decision.should_skip:
                EpubExtractor.append_skipped_section_warning(
                    warnings,
                    significant_warnings,
                    source_chapter,
                    publication_metadata_skip_decision.reason,
                )
                continue

            toc_skip_decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
                readable_spine_index,
                readable_spine_count,
                source_chapter.href,
                source_chapter.title,
                source_chapter.html,
            )
            if toc_skip_decision.should_skip:
                EpubExtractor.append_skipped_section_warning(
                    warnings,
                    significant_warnings,
                    source_chapter,
                    toc_skip_decision.reason,
                )
                continue

            source_chapters.append(source_chapter)

        if not source_chapters:
            warning = "No readable EPUB spine documents found."
            significant_warnings.append(warning)

        return source_chapters, book_title, warnings, significant_warnings

    @staticmethod
    def extract_book_title(book: Any) -> str:
        try:
            metadata_values = book.get_metadata("DC", "title")
        except Exception:
            return ""

        for metadata_value in metadata_values:
            value = metadata_value[0] if isinstance(metadata_value, tuple) and metadata_value else metadata_value
            if not isinstance(value, str):
                continue
            title = BeautifulSoupEpubChapterTextExtractor.normalize_inline_text(value)
            if title:
                return title
        return ""

    @staticmethod
    def append_skipped_section_warning(
            warnings: list[str],
            significant_warnings: list[str],
            source_chapter: EpubSourceChapter,
            reason: str,
    ) -> None:
        warning = EpubExtractor.format_skipped_section_warning(source_chapter, reason)
        warnings.append(warning)
        significant_warnings.append(warning)

    @staticmethod
    def format_skipped_section_warning(source_chapter: EpubSourceChapter, reason: str) -> str:
        preview = EpubExtractor.ellipsize_text(
            EpubSectionSkipDetector.html_to_text_preview(source_chapter.html),
            100,
        )
        return f"Skipped EPUB section: {source_chapter.href} ({reason}): {preview}"

    @staticmethod
    def ellipsize_text(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return "…"[:max_chars]
        return text[:max_chars - 1] + "…"

    @staticmethod
    def decode_item_content(item: Any) -> str:
        try:
            content = item.get_content()
        except Exception as e:
            EpubExtractor.log_warnings([f"Unable to read EPUB item content: {e}"])
            return ""
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def extract_title(html: str, fallback_href: str) -> str:
        BeautifulSoup = BeautifulSoupEpubChapterTextExtractor.import_beautiful_soup()
        soup = BeautifulSoup(html, "html.parser")
        for selector in ["h1", "h2", "title"]:
            tag = soup.find(selector)
            if tag:
                text = BeautifulSoupEpubChapterTextExtractor.normalize_inline_text(tag.get_text(" "))
                if text:
                    return text
        stem = os.path.splitext(os.path.basename(fallback_href))[0]
        return stem or fallback_href

    @staticmethod
    def copy_epub_to_project(epub_path: str, project_dir: str) -> str:
        dest_path = os.path.join(project_dir, PROJECT_TEXT_EPUB_FILE_NAME)
        try:
            shutil.copy(epub_path, dest_path)
            return ""
        except Exception as e:
            message = f"Error saving EPUB copy: {e}"
            L.e(message)
            return message

    @staticmethod
    def log_warnings(warnings: list[str]) -> None:
        for warning in warnings:
            try:
                L.w(warning)
            except Exception:
                pass