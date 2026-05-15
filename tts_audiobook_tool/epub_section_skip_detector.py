from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Any


@dataclass(frozen=True)
class EpubSectionSkipDecision:
    should_skip: bool = False
    reason: str = ""


class EpubSectionSkipDetector:
    """
    Detects EPUB spine sections that should be skipped before audiobook text extraction.

    The detector centralizes section-level skip policy so `EpubExtractor` can remain focused on
    EPUB orchestration. Public detection methods are named by skip reason and return explicit
    decisions where appropriate, making it clear whether a section is skipped because it is a
    navigation document, publication metadata page, or table-of-contents-like page. Strongly
    labeled publication metadata and table-of-contents sections can be skipped in both the front
    and back scan windows, while weaker content-pattern heuristics are limited to front matter to
    reduce false positives.

    These heuristics currently make English-language assumptions. In particular, the publication metadata,
    front/back matter, and table-of-contents keyword lists are English-oriented and should be expanded
    or made language-aware if EPUB import needs robust support for other languages.
    """

    FRONT_MATTER_SKIP_SCAN_LIMIT = 6
    BACK_MATTER_SKIP_SCAN_LIMIT = 4
    CONTENT_ONLY_MAX_WORDS = 700
    CONTENT_ONLY_MAX_CHARS = 4000
    PUBLICATION_METADATA_CONTENT_SIGNAL_THRESHOLD = 2
    TOC_MIN_ANCHORS = 5
    TOC_MIN_LIST_ITEMS = 5
    TOC_MIN_SHORT_LINES = 5
    TOC_SHORT_LINE_MAX_CHARS = 80

    PUBLICATION_METADATA_HREF_TITLE_TERMS = {
        "aboutpublisher",
        "aboutthepublisher",
        "alsoavailable",
        "copyright",
        "copyrgt",
        "colophon",
        "credits",
        "edition",
        "frontmatter",
        "imprint",
        "isbn",
        "legal",
        "license",
        "otherbooks",
        "publisher",
        "publishinghistory",
        "rights",
    }

    TOC_HREF_TITLE_TERMS = {
        "contents",
        "tableofcontents",
        "toc",
    }

    NON_READING_HREF_TITLE_TERMS = {
        "cover",
        "copyright",
        "dedication",
        "halftitle",
        "imprint",
        "landmark",
        "license",
        "nav",
        "title",
        "titlepage",
        "toc",
    }

    NAV_DOCUMENT_BASENAMES = {
        "nav.html",
        "nav.xhtml",
        "toc.html",
        "toc.xhtml",
    }

    PUBLICATION_METADATA_TEXT_SIGNALS = {
        "all rights reserved",
        "copyright",
        "copyrighted material",
        "cover design by",
        "cover image",
        "cover illustration",
        "ebook isbn",
        "edited by",
        "first published",
        "interior design by",
        "isbn",
        "library of congress",
        "no part of this book may be reproduced",
        "printed in",
        "published by",
        "publishing history",
        "this edition",
        "translation copyright",
        "without permission of the publisher",
    }

    @staticmethod
    def is_navigation_document(item_id: str, href: str, item: Any) -> bool:
        if item_id.lower() == "nav":
            return True
        if EpubSectionSkipDetector.basename_lower(href) in EpubSectionSkipDetector.NAV_DOCUMENT_BASENAMES:
            return True
        try:
            properties = getattr(item, "properties", [])
            if "nav" in properties:
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def detect_publication_metadata_skip(
            readable_spine_index: int,
            readable_spine_count: int,
            href: str,
            title: str,
            html: str
    ) -> EpubSectionSkipDecision:
        is_in_scan_window = EpubSectionSkipDetector.is_within_front_or_back_matter_scan_limit(
            readable_spine_index,
            readable_spine_count,
        )

        if EpubSectionSkipDetector.has_publication_metadata_href_or_title_signal(href, title):
            if is_in_scan_window:
                return EpubSectionSkipDecision(True, "publication metadata href or title signal")
            return EpubSectionSkipDecision()

        if not EpubSectionSkipDetector.is_within_front_matter_scan_limit(readable_spine_index):
            return EpubSectionSkipDecision()

        text = EpubSectionSkipDetector.html_to_text_preview(html)
        if not EpubSectionSkipDetector.is_short_enough_for_content_based_publication_metadata_skip(text):
            return EpubSectionSkipDecision()

        signal_count = EpubSectionSkipDetector.count_publication_metadata_text_signals(text)
        if signal_count >= EpubSectionSkipDetector.PUBLICATION_METADATA_CONTENT_SIGNAL_THRESHOLD:
            return EpubSectionSkipDecision(True, f"publication metadata text signals ({signal_count})")

        return EpubSectionSkipDecision()

    @staticmethod
    def detect_table_of_contents_skip(
            readable_spine_index: int,
            readable_spine_count: int,
            href: str,
            title: str,
            html: str
    ) -> EpubSectionSkipDecision:
        has_href_or_title_signal = EpubSectionSkipDetector.has_table_of_contents_href_or_title_signal(href, title)
        has_heading_signal = EpubSectionSkipDetector.has_table_of_contents_heading_signal(html)
        anchor_count = EpubSectionSkipDetector.count_html_anchors(html)
        list_item_count = EpubSectionSkipDetector.count_html_list_items(html)
        short_line_count = EpubSectionSkipDetector.count_short_text_lines(html)
        has_link_structure = anchor_count >= EpubSectionSkipDetector.TOC_MIN_ANCHORS or list_item_count >= EpubSectionSkipDetector.TOC_MIN_LIST_ITEMS
        has_short_line_pattern = short_line_count >= EpubSectionSkipDetector.TOC_MIN_SHORT_LINES
        is_early_spine_section = EpubSectionSkipDetector.is_within_front_matter_scan_limit(readable_spine_index)
        is_in_scan_window = EpubSectionSkipDetector.is_within_front_or_back_matter_scan_limit(
            readable_spine_index,
            readable_spine_count,
        )

        if is_in_scan_window and has_href_or_title_signal and has_link_structure:
            return EpubSectionSkipDecision(True, "table of contents href/title signal plus link structure")
        if is_in_scan_window and has_heading_signal and has_link_structure:
            return EpubSectionSkipDecision(True, "table of contents heading signal plus link structure")
        if is_early_spine_section and anchor_count >= EpubSectionSkipDetector.TOC_MIN_ANCHORS and has_short_line_pattern:
            return EpubSectionSkipDecision(True, "table of contents link density and short-line pattern")

        return EpubSectionSkipDecision()

    @staticmethod
    def is_likely_empty_non_reading_section(href: str, title: str) -> bool:
        text = EpubSectionSkipDetector.normalize_key_text(f"{href} {title}")
        return any(term in text for term in EpubSectionSkipDetector.NON_READING_HREF_TITLE_TERMS)

    @staticmethod
    def is_within_front_matter_scan_limit(readable_spine_index: int) -> bool:
        return readable_spine_index < EpubSectionSkipDetector.FRONT_MATTER_SKIP_SCAN_LIMIT

    @staticmethod
    def is_within_back_matter_scan_limit(readable_spine_index: int, readable_spine_count: int) -> bool:
        return readable_spine_index >= max(0, readable_spine_count - EpubSectionSkipDetector.BACK_MATTER_SKIP_SCAN_LIMIT)

    @staticmethod
    def is_within_front_or_back_matter_scan_limit(readable_spine_index: int, readable_spine_count: int) -> bool:
        return (
            EpubSectionSkipDetector.is_within_front_matter_scan_limit(readable_spine_index)
            or EpubSectionSkipDetector.is_within_back_matter_scan_limit(readable_spine_index, readable_spine_count)
        )

    @staticmethod
    def has_publication_metadata_href_or_title_signal(href: str, title: str) -> bool:
        text = EpubSectionSkipDetector.normalize_key_text(f"{href} {title}")
        return any(term in text for term in EpubSectionSkipDetector.PUBLICATION_METADATA_HREF_TITLE_TERMS)

    @staticmethod
    def has_table_of_contents_href_or_title_signal(href: str, title: str) -> bool:
        text = EpubSectionSkipDetector.normalize_key_text(f"{href} {title}")
        return any(term in text for term in EpubSectionSkipDetector.TOC_HREF_TITLE_TERMS)

    @staticmethod
    def has_table_of_contents_heading_signal(html: str) -> bool:
        heading_text = " ".join(re.findall(r"(?is)<(?:h1|h2|title|strong|b)[^>]*>(.*?)</(?:h1|h2|title|strong|b)>", html)[:4])
        heading_text = EpubSectionSkipDetector.html_to_text_preview(heading_text)
        return EpubSectionSkipDetector.has_table_of_contents_text_signal(heading_text)

    @staticmethod
    def has_table_of_contents_text_signal(text: str) -> bool:
        normalized = EpubSectionSkipDetector.normalize_key_text(text)
        return any(term in normalized for term in EpubSectionSkipDetector.TOC_HREF_TITLE_TERMS)

    @staticmethod
    def count_publication_metadata_text_signals(text: str) -> int:
        normalized = EpubSectionSkipDetector.normalize_content_text(text)
        return sum(1 for signal in EpubSectionSkipDetector.PUBLICATION_METADATA_TEXT_SIGNALS if signal in normalized)

    @staticmethod
    def is_short_enough_for_content_based_publication_metadata_skip(text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        word_count = len(re.findall(r"\w+", text))
        return word_count <= EpubSectionSkipDetector.CONTENT_ONLY_MAX_WORDS or len(text) <= EpubSectionSkipDetector.CONTENT_ONLY_MAX_CHARS

    @staticmethod
    def count_html_anchors(html: str) -> int:
        return len(re.findall(r"(?is)<a\b[^>]*\bhref\s*=", html))

    @staticmethod
    def count_html_list_items(html: str) -> int:
        return len(re.findall(r"(?is)<li\b", html))

    @staticmethod
    def count_short_text_lines(html: str) -> int:
        lines = [line.strip() for line in EpubSectionSkipDetector.html_to_text_preview(html).splitlines()]
        lines = [line for line in lines if line]
        return sum(1 for line in lines if len(line) <= EpubSectionSkipDetector.TOC_SHORT_LINE_MAX_CHARS)

    @staticmethod
    def html_to_text_preview(html: str) -> str:
        text = re.sub(r"(?is)<(script|style|head|nav|svg)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</(p|div|section|article|h[1-6]|li|tr|blockquote|a)>", "\n", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = unescape(text).replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    def normalize_key_text(text: str) -> str:
        text = unescape(text).lower()
        return re.sub(r"[^a-z0-9]+", "", text)

    @staticmethod
    def normalize_content_text(text: str) -> str:
        text = unescape(text).lower().replace("\xa0", " ")
        text = text.replace("©", " copyright ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def basename_lower(path: str) -> str:
        path = path.replace("\\", "/")
        return path.rsplit("/", 1)[-1].lower()
