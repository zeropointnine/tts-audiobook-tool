from __future__ import annotations

import os
import posixpath
import re
import shutil
import importlib
from dataclasses import dataclass, field
from html import unescape
from urllib.parse import unquote, urlsplit
from typing import Any, Protocol

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.constants import PROJECT_TEXT_EPUB_FILE_NAME
from tts_audiobook_tool.text_ops.epub_section_skip_detector import EpubSectionSkipDetector
from tts_audiobook_tool.l import L
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


# The legacy constant name remains the compatibility switch even though boundaries are now logical.
DOWNGRADE_LEADING_SECTIONS_AFTER_EPUB_BOUNDARY = True


@dataclass(frozen=True)
class EpubNavigationTarget:
    title: str
    path: str
    fragment: str
    href: str
    order: int
    depth: int = 0


@dataclass
class EpubSourceChapter:
    title: str
    href: str
    media_type: str
    html: str
    navigation_targets: list[EpubNavigationTarget] = field(default_factory=list)


@dataclass
class EpubTextChapter:
    title: str
    href: str
    text: str


@dataclass
class EpubTextSlice:
    text: str
    navigation_target: EpubNavigationTarget | None = None


@dataclass
class EpubTextExtractionResult:
    text: str
    warnings: list[str] = field(default_factory=list)
    significant_warnings: list[str] = field(default_factory=list)
    slices: list[EpubTextSlice] = field(default_factory=list)
    has_unresolved_navigation_target: bool = False


@dataclass
class EpubTextExtractionStats:
    inline_whitespace_repairs: int = 0


@dataclass
class EpubImportResult:
    phrase_groups: list[PhraseGroup]
    raw_text: str
    section_start_indices: list[int]
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
        target_fragments = {
            target.fragment
            for target in chapter.navigation_targets
            if target.fragment
        }
        anchor_piece_indices: dict[str, int] = {}

        root = soup.body or soup
        pieces: list[str] = []
        self.append_node_text(
            root,
            pieces,
            stats,
            target_fragments=target_fragments,
            anchor_piece_indices=anchor_piece_indices,
        )
        text = self.normalize_output_text("".join(pieces))
        slices, navigation_warnings, has_unresolved_navigation_target = self.make_text_slices(
            chapter,
            pieces,
            anchor_piece_indices,
        )
        warnings.extend(navigation_warnings)
        significant_warnings.extend(navigation_warnings)

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
            slices=slices,
            has_unresolved_navigation_target=has_unresolved_navigation_target,
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

    def append_node_text(
            self,
            node: Any,
            pieces: list[str],
            stats: EpubTextExtractionStats,
            target_fragments: set[str] | None = None,
            anchor_piece_indices: dict[str, int] | None = None,
    ) -> None:
        name = getattr(node, "name", None)

        if target_fragments and anchor_piece_indices is not None and hasattr(node, "get"):
            anchor_values = [node.get("id")]
            if name == "a":
                anchor_values.append(node.get("name"))
            for anchor_value in anchor_values:
                if anchor_value in target_fragments and anchor_value not in anchor_piece_indices:
                    anchor_piece_indices[anchor_value] = len(pieces)

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
                    self.append_node_text(
                        child,
                        pieces,
                        stats,
                        target_fragments=target_fragments,
                        anchor_piece_indices=anchor_piece_indices,
                    )

        if is_block:
            pieces.append("\n\n")

    @classmethod
    def make_text_slices(
            cls,
            chapter: EpubSourceChapter,
            pieces: list[str],
            anchor_piece_indices: dict[str, int],
    ) -> tuple[list[EpubTextSlice], list[str], bool]:
        warnings: list[str] = []
        has_unresolved_navigation_target = False
        targets_by_piece_index: dict[int, EpubNavigationTarget] = {}

        for target in chapter.navigation_targets:
            if target.fragment:
                piece_index = anchor_piece_indices.get(target.fragment)
                if piece_index is None:
                    warnings.append(
                        f"EPUB navigation target not found in {chapter.href}: {target.href}"
                    )
                    has_unresolved_navigation_target = True
                    continue
            else:
                piece_index = 0

            previous = targets_by_piece_index.get(piece_index)
            if (
                previous is None
                or target.depth > previous.depth
                or (target.depth == previous.depth and target.order < previous.order)
            ):
                targets_by_piece_index[piece_index] = target

        targets_in_nav_order = sorted(
            targets_by_piece_index.items(),
            key=lambda item: item[1].order,
        )
        accepted_boundaries: list[tuple[int, EpubNavigationTarget]] = []
        last_piece_index = -1
        for piece_index, target in targets_in_nav_order:
            if piece_index < last_piece_index:
                warnings.append(
                    f"EPUB navigation target is out of reading order in {chapter.href}: {target.href}"
                )
                continue
            accepted_boundaries.append((piece_index, target))
            last_piece_index = piece_index

        accepted_boundaries.sort(key=lambda item: item[0])
        slices: list[EpubTextSlice] = []
        start_piece_index = 0
        current_target: EpubNavigationTarget | None = None
        for piece_index, target in accepted_boundaries:
            raw_text = "".join(pieces[start_piece_index:piece_index])
            normalized_text = cls.normalize_output_text(raw_text)
            if normalized_text or current_target is not None:
                slices.append(EpubTextSlice(normalized_text, current_target))
            current_target = target
            start_piece_index = piece_index

        raw_text = "".join(pieces[start_piece_index:])
        normalized_text = cls.normalize_output_text(raw_text)
        if normalized_text or current_target is not None or not slices:
            slices.append(EpubTextSlice(normalized_text, current_target))
        return slices, warnings, has_unresolved_navigation_target

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
    def get_ebook_title(epub_path: str) -> str:
        _, book_title, _, _ = EpubExtractor.load_source_chapters(epub_path)
        return book_title

    @staticmethod
    def import_epub(
            epub_path: str,
            max_words: int,
            segmentation_strategy: SegmentationStrategy,
            language_code: str,
            dialog_segmentation: bool = False,
            extractor: EpubChapterTextExtractor | None = None
    ) -> EpubImportResult:
        source_chapters, book_title, warnings, significant_warnings = EpubExtractor.load_source_chapters(epub_path)
        EpubExtractor.log_warnings(warnings)
        EpubExtractor.log_warnings([warning for warning in significant_warnings if warning not in warnings])
        extractor = extractor or EpubExtractor.DEFAULT_EXTRACTOR

        extracted_chapters: list[tuple[EpubSourceChapter, EpubTextExtractionResult]] = []
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
            extracted_chapters.append((source_chapter, result))

        text_chapters = EpubExtractor.assemble_text_chapters(
            extracted_chapters,
            warnings,
            significant_warnings,
        )

        phrase_groups: list[PhraseGroup] = []
        markers: list[int] = []
        raw_text_parts: list[str] = []
        kept_chapters: list[EpubTextChapter] = []

        for chapter in text_chapters:
            chapter_text = chapter.text
            if not chapter_text.strip():
                continue
            if not app_text.is_vocalizable(chapter_text):
                # Ornament-only pages (a lone dinkus, divider, etc) contain no
                # speakable text. Keeping them would force a section whose
                # groups are all non-vocalizable, so skip them outright.
                warning = (
                    "EPUB section has no vocalizable text and was skipped: "
                    f"{chapter.title or chapter.href}"
                )
                warnings.append(warning)
                significant_warnings.append(warning)
                EpubExtractor.log_warnings([warning])
                continue
            chapter_phrase_groups = PhraseGrouper.text_to_groups(
                chapter_text,
                max_words=max_words,
                strategy=segmentation_strategy,
                pysbd_lang=language_code,
                dialog_segmentation=dialog_segmentation,
            )
            if phrase_groups:
                markers.append(len(phrase_groups))
            if phrase_groups and DOWNGRADE_LEADING_SECTIONS_AFTER_EPUB_BOUNDARY:
                EpubExtractor.downgrade_leading_section_groups(chapter_phrase_groups)
            EpubExtractor.mark_last_phrase_as_section(chapter_phrase_groups)
            phrase_groups.extend(chapter_phrase_groups)
            raw_text_parts.append(chapter_text)
            kept_chapters.append(chapter)

        raw_text = "\n\n".join(raw_text_parts)

        if not phrase_groups:
            warning = "EPUB import produced no text segments."
            warnings.append(warning)
            significant_warnings.append(warning)
            L.w(warning)

        return EpubImportResult(
            phrase_groups=phrase_groups,
            raw_text=raw_text,
            section_start_indices=markers,
            chapters=kept_chapters,
            book_title=book_title,
            warnings=warnings,
            significant_warnings=significant_warnings,
        )

    @staticmethod
    def assemble_text_chapters(
            extracted_chapters: list[tuple[EpubSourceChapter, EpubTextExtractionResult]],
            warnings: list[str],
            significant_warnings: list[str],
    ) -> list[EpubTextChapter]:
        has_navigation_targets = any(
            source_chapter.navigation_targets
            for source_chapter, _ in extracted_chapters
        )
        has_unresolved_navigation_target = any(
            result.has_unresolved_navigation_target
            for _, result in extracted_chapters
        )
        if has_navigation_targets and not has_unresolved_navigation_target:
            text_chapters, used_navigation = EpubExtractor.assemble_navigation_text_chapters(
                extracted_chapters,
                warnings,
                significant_warnings,
            )
            if used_navigation:
                return text_chapters

        if extracted_chapters:
            warning = "EPUB navigation is unavailable or unusable; section boundaries were derived from spine documents."
            warnings.append(warning)
            significant_warnings.append(warning)
            EpubExtractor.log_warnings([warning])
        return EpubExtractor.make_spine_text_chapters(extracted_chapters)

    @staticmethod
    def make_spine_text_chapters(
            extracted_chapters: list[tuple[EpubSourceChapter, EpubTextExtractionResult]],
    ) -> list[EpubTextChapter]:
        return [
            EpubTextChapter(
                title=source_chapter.title,
                href=source_chapter.href,
                text=result.text,
            )
            for source_chapter, result in extracted_chapters
            if result.text.strip()
        ]

    @staticmethod
    def assemble_navigation_text_chapters(
            extracted_chapters: list[tuple[EpubSourceChapter, EpubTextExtractionResult]],
            warnings: list[str],
            significant_warnings: list[str],
    ) -> tuple[list[EpubTextChapter], bool]:
        text_chapters: list[EpubTextChapter] = []
        text_parts: list[str] = []
        current_target: EpubNavigationTarget | None = None
        fallback_title = ""
        fallback_href = ""
        used_navigation = False

        def flush_text_chapter() -> None:
            nonlocal text_parts, current_target, fallback_title, fallback_href, used_navigation
            if not text_parts:
                return
            # Merged pseudo-sections keep a section-like boundary: two blank lines between parts.
            title = current_target.title if current_target is not None else fallback_title
            href = current_target.href if current_target is not None else fallback_href
            text_chapters.append(EpubTextChapter(title=title, href=href, text="\n\n\n".join(text_parts)))
            used_navigation = used_navigation or current_target is not None
            text_parts = []
            current_target = None
            fallback_title = ""
            fallback_href = ""

        for source_chapter, result in extracted_chapters:
            if result.slices:
                slices = result.slices
            else:
                document_targets = [
                    target
                    for target in source_chapter.navigation_targets
                    if not target.fragment
                ]
                document_target = max(document_targets, key=lambda target: target.order, default=None)
                slices = [EpubTextSlice(result.text, document_target)]
            for text_slice in slices:
                if text_slice.navigation_target is not None:
                    if text_parts:
                        flush_text_chapter()
                    elif current_target is not None:
                        warning = (
                            f"Omitted empty EPUB navigation section before {text_slice.navigation_target.href}: "
                            f"{current_target.title}"
                        )
                        warnings.append(warning)
                        significant_warnings.append(warning)
                    current_target = text_slice.navigation_target

                if text_slice.text.strip():
                    if current_target is None and not text_parts:
                        fallback_title = source_chapter.title
                        fallback_href = source_chapter.href
                    text_parts.append(text_slice.text)

            if (
                not result.text.strip()
                and current_target is not None
                and current_target.path == EpubExtractor.normalize_toc_href(source_chapter.href)
                and EpubSectionSkipDetector.is_likely_empty_non_reading_section(
                    source_chapter.href,
                    current_target.title,
                )
            ):
                current_target = None

        if text_parts:
            flush_text_chapter()
        elif current_target is not None:
            warning = f"Omitted empty EPUB navigation section: {current_target.title}"
            warnings.append(warning)
            significant_warnings.append(warning)

        return text_chapters, used_navigation

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
    def mark_last_phrase_as_section(phrase_groups: list[PhraseGroup]) -> None:
        if not phrase_groups:
            return
        last_group = phrase_groups[-1]
        if not last_group.phrases:
            return
        last_phrase = last_group.phrases[-1]
        last_phrase.reason = Reason.SECTION_BREAK

    @staticmethod
    def downgrade_leading_section_groups(phrase_groups: list[PhraseGroup]) -> None:
        """
        Downgrades section-like groups at the start of an EPUB logical section.

        EPUB import force-marks the previous logical section's final phrase as a section
        break. If the next section begins with heading text that also ends in section-like
        spacing, that leading source-derived break is redundant and should behave as a
        paragraph. The navigation boundary remains on the previous section.
        """

        for group in phrase_groups:
            if group.last_reason != Reason.SPACE_BREAK:
                break
            last_phrase = group.phrases[-1]
            last_phrase.reason = Reason.PARAGRAPH
            last_phrase.text = last_phrase.text.rstrip() + "\n\n"

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
            book = epub.read_epub(epub_path, options={"ignore_ncx": True})
        except Exception as e:
            message = f"Error reading EPUB: {e}"
            L.e(message)
            return [], "", [], [message]

        book_title = EpubExtractor.extract_book_title(book)
        navigation_targets = EpubExtractor.extract_navigation_targets(book)
        navigation_targets_by_path: dict[str, list[EpubNavigationTarget]] = {}
        for navigation_target in navigation_targets:
            navigation_targets_by_path.setdefault(navigation_target.path, []).append(navigation_target)
        warnings: list[str] = []
        significant_warnings: list[str] = []
        source_chapter_candidates: list[EpubSourceChapter] = []
        source_chapters: list[EpubSourceChapter] = []
        # Scan-window position that only counts text-bearing documents, so image-only cover and
        # insert pages do not consume front/back matter scan budget.
        reading_spine_position = 0

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

            item_navigation_targets = navigation_targets_by_path.get(EpubExtractor.normalize_toc_href(href), [])
            title = (
                item_navigation_targets[0].title
                if item_navigation_targets
                else EpubExtractor.extract_title(html, href)
            )
            if EpubSectionSkipDetector.is_navigation_document(str(item_id), href, item):
                toc_skip_decision = EpubSectionSkipDetector.detect_table_of_contents_skip(
                    readable_spine_index=reading_spine_position,
                    readable_spine_count=max(1, reading_spine_position + 1),
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
                navigation_targets=list(item_navigation_targets),
            ))
            if EpubSectionSkipDetector.is_text_bearing_spine_document(html):
                reading_spine_position += 1

        candidate_paths = {
            EpubExtractor.normalize_toc_href(source_chapter.href)
            for source_chapter in source_chapter_candidates
        }
        for navigation_target in navigation_targets:
            if navigation_target.path in candidate_paths:
                continue
            warning = f"EPUB navigation target does not map to a spine document: {navigation_target.href}"
            warnings.append(warning)
            significant_warnings.append(warning)

        EpubExtractor.remove_backward_document_navigation_targets(
            source_chapter_candidates,
            warnings,
            significant_warnings,
        )
        # Scan-window positions that image-only documents do not consume, so front/back matter is
        # still recognized when many illustration pages precede or follow it.
        text_bearing_flags = [
            EpubSectionSkipDetector.is_text_bearing_spine_document(source_chapter.html)
            for source_chapter in source_chapter_candidates
        ]
        reading_spine_count = sum(text_bearing_flags)
        reading_spine_position = 0
        for source_chapter, is_text_bearing in zip(source_chapter_candidates, text_bearing_flags):
            publication_metadata_skip_decision = EpubSectionSkipDetector.detect_publication_metadata_skip(
                reading_spine_position,
                reading_spine_count,
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
                reading_spine_position,
                reading_spine_count,
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
            if is_text_bearing:
                reading_spine_position += 1

        if not source_chapters:
            warning = "No readable EPUB spine documents found."
            significant_warnings.append(warning)

        return source_chapters, book_title, warnings, significant_warnings

    @staticmethod
    def get_navigation_resource_base_path(book: Any) -> str:
        try:
            items = getattr(book, "items", [])
        except Exception:
            return ""

        for item in items:
            properties = getattr(item, "properties", [])
            if "nav" in properties or type(item).__name__ == "EpubNav":
                # EbookLib has already resolved EPUB 3 nav links relative to this resource.
                return ""

        for item in items:
            if getattr(item, "media_type", "") != "application/x-dtbncx+xml":
                continue
            file_name = EpubExtractor.normalize_toc_href(getattr(item, "file_name", ""))
            if file_name:
                directory = posixpath.dirname(file_name)
                return "" if directory == "." else directory
        return ""

    @staticmethod
    def extract_navigation_targets(book: Any) -> list[EpubNavigationTarget]:
        try:
            toc = getattr(book, "toc", [])
        except Exception:
            return []

        navigation_targets: list[EpubNavigationTarget] = []
        navigation_base_path = EpubExtractor.get_navigation_resource_base_path(book)
        EpubExtractor.collect_navigation_targets(
            toc,
            navigation_targets,
            navigation_base_path,
        )
        return navigation_targets

    @staticmethod
    def collect_navigation_targets(
            toc_items: Any,
            navigation_targets: list[EpubNavigationTarget],
            navigation_base_path: str = "",
            depth: int = 0,
    ) -> None:
        if toc_items is None:
            return

        if isinstance(toc_items, tuple):
            for tuple_index, toc_item in enumerate(toc_items):
                EpubExtractor.collect_navigation_targets(
                    toc_item,
                    navigation_targets,
                    navigation_base_path,
                    depth + (1 if tuple_index > 0 else 0),
                )
            return

        if isinstance(toc_items, list):
            for toc_item in toc_items:
                EpubExtractor.collect_navigation_targets(
                    toc_item,
                    navigation_targets,
                    navigation_base_path,
                    depth,
                )
            return

        href = getattr(toc_items, "href", "")
        title = BeautifulSoupEpubChapterTextExtractor.normalize_inline_text(getattr(toc_items, "title", ""))
        navigation_target = EpubExtractor.make_navigation_target(
            href,
            title,
            len(navigation_targets),
            navigation_base_path,
            depth,
        )
        if navigation_target is not None:
            navigation_targets.append(navigation_target)

        subitems = getattr(toc_items, "subitems", None)
        if subitems:
            EpubExtractor.collect_navigation_targets(
                subitems,
                navigation_targets,
                navigation_base_path,
                depth + 1,
            )

    @staticmethod
    def make_navigation_target(
            href: str,
            title: str,
            order: int,
            navigation_base_path: str = "",
            depth: int = 0,
    ) -> EpubNavigationTarget | None:
        if not isinstance(href, str) or not title:
            return None
        parts = urlsplit(href)
        if parts.scheme or parts.netloc:
            return None
        path_href = parts.path
        if navigation_base_path and not path_href.startswith("/"):
            path_href = posixpath.join(navigation_base_path, path_href)
        path = EpubExtractor.normalize_toc_href(path_href)
        if not path:
            return None
        fragment = unquote(parts.fragment).strip()
        canonical_href = path + (f"#{fragment}" if fragment else "")
        return EpubNavigationTarget(
            title=title,
            path=path,
            fragment=fragment,
            href=canonical_href,
            order=order,
            depth=depth,
        )

    @staticmethod
    def extract_toc_title_by_href(book: Any) -> dict[str, str]:
        title_by_href: dict[str, str] = {}
        for target in EpubExtractor.extract_navigation_targets(book):
            title_by_href.setdefault(target.path, target.title)
        return title_by_href

    @staticmethod
    def collect_toc_titles(toc_items: Any, title_by_href: dict[str, str]) -> None:
        navigation_targets: list[EpubNavigationTarget] = []
        EpubExtractor.collect_navigation_targets(toc_items, navigation_targets)
        for target in navigation_targets:
            title_by_href.setdefault(target.path, target.title)

    @staticmethod
    def normalize_toc_href(href: str) -> str:
        if not isinstance(href, str):
            return ""
        parts = urlsplit(href)
        if parts.scheme or parts.netloc:
            return ""
        path = unquote(parts.path).replace("\\", "/").strip()
        if not path:
            return ""
        path = posixpath.normpath(path)
        while path.startswith("./"):
            path = path[2:]
        return path.lstrip("/")

    @staticmethod
    def remove_backward_document_navigation_targets(
            source_chapters: list[EpubSourceChapter],
            warnings: list[str],
            significant_warnings: list[str],
    ) -> None:
        spine_index_by_path = {
            EpubExtractor.normalize_toc_href(source_chapter.href): index
            for index, source_chapter in enumerate(source_chapters)
        }
        all_targets = sorted(
            (
                target
                for source_chapter in source_chapters
                for target in source_chapter.navigation_targets
            ),
            key=lambda target: target.order,
        )
        rejected_orders: set[int] = set()
        last_spine_index = -1
        for target in all_targets:
            spine_index = spine_index_by_path.get(target.path)
            if spine_index is None:
                continue
            if spine_index < last_spine_index:
                warning = f"EPUB navigation target is out of spine order: {target.href}"
                warnings.append(warning)
                significant_warnings.append(warning)
                rejected_orders.add(target.order)
                continue
            last_spine_index = spine_index

        if not rejected_orders:
            return
        for source_chapter in source_chapters:
            source_chapter.navigation_targets = [
                target
                for target in source_chapter.navigation_targets
                if target.order not in rejected_orders
            ]

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
        if EpubExtractor.paths_reference_same_file(epub_path, dest_path):
            # Re-importing the saved project EPUB copy; it is already in place
            return ""
        try:
            shutil.copy(epub_path, dest_path)
            return ""
        except Exception as e:
            message = f"Error saving EPUB copy: {e}"
            L.e(message)
            return message

    @staticmethod
    def paths_reference_same_file(path_a: str, path_b: str) -> bool:
        if os.path.abspath(path_a) == os.path.abspath(path_b):
            return True
        try:
            return os.path.samefile(path_a, path_b)
        except OSError:
            return False

    @staticmethod
    def log_warnings(warnings: list[str]) -> None:
        for warning in warnings:
            try:
                L.w(warning)
            except Exception:
                pass