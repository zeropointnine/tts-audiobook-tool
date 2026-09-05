from __future__ import annotations
import math
import string
import re
import unicodedata
from typing import Protocol, cast
import pysbd

from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.text_ops.quote_spans import find_quote_spans


DOWNGRADE_CONSECUTIVE_SECTIONS = True


class _PysbdTextSpan(Protocol):
    sent: str
    start: int


class PhraseSegmenter:
    """
    Creates Phrases from strings
    """

    @staticmethod
    def text_to_phrases(text: str, max_words: int, pysbd_lang: str) -> list[Phrase]:
        """
        Returns list of Phrases with 'reasons', ready to be grouped as needed
        """
        phrases: list[Phrase] = []

        # Text to sentence strings
        sentence_strings = PhraseSegmenter.string_to_sentence_strings(text, pysbd_lang)

        for sentence_string in sentence_strings:

            # Sentence strings to phrase strings
            phrase_strings = PhraseSegmenter.sentence_string_to_phrase_strings(sentence_string, pysbd_lang)

            # Make Phrases proper, disambiguating btw sentence/paragraph/section
            for i, phrase in enumerate(phrase_strings):
                is_last_phrase_in_sentence = (i == len(phrase_strings) - 1)
                if is_last_phrase_in_sentence:
                    num_lf = app_text.num_trailing_line_breaks(phrase)
                    match num_lf:
                        case 0:
                            reason = Reason.SENTENCE
                        case 1:
                            reason = Reason.PARAGRAPH
                        case 2:
                            reason = Reason.PARAGRAPH
                        case _: # >= 3
                            reason = Reason.SPACE_BREAK
                else:
                    reason = Reason.PHRASE
                phrases.append( Phrase(phrase, reason) )

        # Split long phrases if necessary
        new_result: list[Phrase] = []
        for phrase in phrases:
            phrases = PhraseSegmenter.long_phrase_to_phrases(phrase, max_words)
            new_result.extend(phrases)
        phrases = new_result

        phrases = PhraseSegmenter.merge_ornamental_lines(phrases)
        if DOWNGRADE_CONSECUTIVE_SECTIONS:
            phrases = PhraseSegmenter.downgrade_consecutive_space_breaks(phrases)

        return phrases

    @staticmethod
    def string_to_sentence_strings(source: str, pysbd_lang: str) -> list[str]:
        """Segment source into sentences while preserving every character.

        pySBD deliberately protects punctuation inside balanced quotations. We
        retain its full-source boundaries, then collect additional boundaries
        from each balanced quote interior and finally slice the original text.
        """

        from pysbd.languages import Language
        try:
            _ = Language.get_language_code(pysbd_lang)
        except Exception:
            pysbd_lang = "en"  # fail silently

        if not source:
            return []

        boundaries = set(
            PhraseSegmenter._pysbd_boundary_offsets(source, pysbd_lang)
        )
        quote_spans = find_quote_spans(source)

        # pySBD can place a boundary immediately before a closing quote. Move
        # such boundaries past the delimiter and its boundary whitespace, or
        # suppress them before a lowercase continuation/attribution.
        for span in quote_spans:
            closing_quote_offset = span.end - 1
            if closing_quote_offset not in boundaries:
                continue
            boundaries.remove(closing_quote_offset)
            adjusted = PhraseSegmenter._boundary_after_closing_quote(
                source,
                span.end,
            )
            if PhraseSegmenter._starts_new_sentence(source, adjusted):
                boundaries.add(adjusted)

        for span in quote_spans:
            interior_start = span.start + 1
            interior_end = span.end - 1
            interior = source[interior_start:interior_end]
            for offset in PhraseSegmenter._pysbd_boundary_offsets(
                interior,
                pysbd_lang,
            ):
                # The quote interior's final boundary is represented by its
                # closing delimiter and the outer pySBD pass. Adding it here
                # would detach a following attribution such as `she said.`.
                if offset < len(interior):
                    boundaries.add(interior_start + offset)

        # Add a missing outer boundary after a terminally punctuated quote when
        # pySBD does not understand that delimiter style. Lowercase text remains
        # attached as a likely continuation/attribution.
        for span in quote_spans:
            interior = source[span.start + 1:span.end - 1].rstrip()
            adjusted = PhraseSegmenter._boundary_after_closing_quote(
                source,
                span.end,
            )
            if (
                interior
                and interior[-1] in ".!?。．！？؟።፧۔։՜"
                and PhraseSegmenter._starts_new_sentence(source, adjusted)
            ):
                boundaries.add(adjusted)

        # pySBD commonly protects adjacent quote spans as one unit. Their
        # separating whitespace belongs to the preceding sentence.
        for previous, following in zip(quote_spans, quote_spans[1:]):
            if (
                previous.end <= following.start
                and source[previous.end:following.start].isspace()
            ):
                boundaries.add(following.start)

        usable_boundaries = sorted(
            offset for offset in boundaries if 0 < offset < len(source)
        )
        sentences: list[str] = []
        start = 0
        for end in usable_boundaries:
            sentences.append(source[start:end])
            start = end
        sentences.append(source[start:])

        return PhraseSegmenter._merge_dangling_punc_words(sentences)

    @staticmethod
    def _pysbd_boundary_offsets(source: str, pysbd_lang: str) -> list[int]:
        """Return logical pySBD sentence starts after the first sentence.

        pySBD spans can overlap or leave gaps around ellipses. Starts of the
        surviving logical spans are therefore safer boundaries than ends of
        their predecessors; slicing later assigns every gap character exactly
        once. Punctuation-only spans are folded into the prior logical span.
        """
        if not source:
            return []
        segmenter = pysbd.Segmenter(
            language=pysbd_lang,
            clean=False,
            char_span=True,
        )
        logical_spans: list[_PysbdTextSpan] = []
        spans = cast(list[_PysbdTextSpan], segmenter.segment(source))
        for span in spans:
            if logical_spans and app_text.is_ws_punc(span.sent):
                continue
            logical_spans.append(span)
        return [span.start for span in logical_spans[1:]]

    @staticmethod
    def _boundary_after_closing_quote(source: str, offset: int) -> int:
        """Advance over punctuation and whitespace attached to a close quote."""
        while (
            offset < len(source)
            and unicodedata.category(source[offset]).startswith("P")
            and source[offset] not in {'"', "'", "“", "‘", "«", "‹", "「", "『"}
        ):
            offset += 1
        while offset < len(source) and source[offset] in string.whitespace:
            offset += 1
        return offset

    @staticmethod
    def _starts_new_sentence(source: str, offset: int) -> bool:
        """Use pySBD's conservative capital/open-quote continuation signal."""
        if offset >= len(source):
            return False
        char = source[offset]
        return char.isupper() or char in {'"', "'", "“", "‘", "«", "‹", "「", "『"}

    @staticmethod
    def _merge_dangling_punc_words(sentences: list[str]) -> list[str]:
        """Attach whitespace/punctuation-only sentence fragments backward."""
        result: list[str] = []
        for sentence in sentences:
            if result and app_text.is_ws_punc(sentence):
                result[-1] += sentence
            else:
                result.append(sentence)
        return result

    @staticmethod
    def sentence_string_to_phrase_strings(sentence: str, language_code: str) -> list[str]:
        """
        Returns list of phrase strings from a sentence string
        """

        # Split after:
        # - double-quote+space `" `
        # - fancy-close-double-quote+space `” `
        # - comma+space `, `
        # - semicolon+space `; `
        # - colon+space `: `
        # - en-dash `–`
        # - em-dash `—`
        # - double normal dash `--` (see double_dash_break_offsets)
        #
        # Parenthetical boundaries are selected separately because whether they
        # merit a spoken phrase depends on their matched content.

        # Using lookbehind for splits after pattern
        pattern = r'(?<=" )|(?<=” )|(?<=, )|(?<=; )|(?<=: )|(?<=–)|(?<=—)'

        # Collect every boundary as a source offset. Ordinary punctuation inside
        # a short or reference-like parenthetical is suppressed too; otherwise a
        # citation comma could recreate the very fragment this policy avoids.
        protected_parentheticals = PhraseSegmenter.non_phrase_parenthetical_ranges(sentence, language_code)
        punctuation_offsets = [match.start() for match in re.finditer(pattern, sentence)]
        ordinary_offsets = (
            punctuation_offsets
            + PhraseSegmenter.double_dash_break_offsets(sentence)
            + PhraseSegmenter.ellipsis_phrase_break_offsets(sentence)
        )
        ordinary_offsets = [
            offset
            for offset in ordinary_offsets
            if not any(open_index < offset <= close_index for open_index, close_index in protected_parentheticals)
        ]
        break_offsets = sorted(set(
            ordinary_offsets
            + PhraseSegmenter.parenthetical_phrase_break_offsets(sentence, language_code)
        ))
        items: list[str] = []
        start = 0
        for offset in break_offsets:
            items.append(sentence[start:offset])
            start = offset
        items.append(sentence[start:])

        # Not including these punctuation characters on purpose:
        # single normal dash, apostrophe/single-quote or double-quote

        # Remove empty strings (e.g. if sentence ends with a delimiter)
        items = [item for item in items if item]

        if len(items) <= 1:
            return items

        def split_leading_ws(input_string: str) -> tuple[str, str]:
            for i, char in enumerate(input_string):
                if char not in string.whitespace:
                    return (input_string[:i], input_string[i:])
            return (input_string, "")

        # Move leading ws_punc to end of previous item
        for i in range(1, len(items)):
            leading_ws_punc, remainder = split_leading_ws(items[i])
            items[i - 1] += leading_ws_punc
            items[i] = remainder

        # Combine any all-ws/punc items to predecessor (edge case)
        new_items = [ items[0] ]
        for i in range(1, len(items)):
            if app_text.is_ws_punc(items[i]):
                new_items[len(new_items) - 1] += items[i]
            else:
                new_items.append(items[i])
        items = new_items
        return items

    @staticmethod
    def parenthetical_phrase_break_offsets(sentence: str, language_code: str) -> list[int]:
        """Return boundaries around substantial, balanced parenthetical asides.

        Parentheses alone are too weak a signal for a spoken phrase: short labels,
        acronyms, citations, and references usually sound better attached to their
        surrounding text. Isolating only parentheticals with at least three
        vocalizable words gives substantial asides a natural pause without
        fragmenting those common compact uses.
        """
        pairs = PhraseSegmenter.balanced_top_level_parenthetical_pairs(sentence)
        if pairs is None:
            return []

        offsets: list[int] = []
        for open_index, close_index in pairs:
            content = sentence[open_index + 1:close_index]
            if PhraseSegmenter.is_phrase_worthy_parenthetical(content, language_code):
                offsets.extend((open_index, close_index + 1))

        return offsets

    @staticmethod
    def non_phrase_parenthetical_ranges(sentence: str, language_code: str) -> list[tuple[int, int]]:
        pairs = PhraseSegmenter.balanced_top_level_parenthetical_pairs(sentence)
        if pairs is None:
            return []
        return [
            (open_index, close_index)
            for open_index, close_index in pairs
            if not PhraseSegmenter.is_phrase_worthy_parenthetical(
                sentence[open_index + 1:close_index],
                language_code,
            )
        ]

    @staticmethod
    def balanced_top_level_parenthetical_pairs(sentence: str) -> list[tuple[int, int]] | None:
        stack: list[int] = []
        pairs: list[tuple[int, int]] = []

        for index, char in enumerate(sentence):
            if char == "(":
                stack.append(index)
            elif char == ")":
                if not stack:
                    return None
                open_index = stack.pop()
                if not stack:
                    pairs.append((open_index, index))

        return None if stack else pairs

    @staticmethod
    def is_phrase_worthy_parenthetical(content: str, language_code: str) -> bool:
        return (
            app_text.get_word_count(content, vocalizable_only=True) >= 3
            and not PhraseSegmenter.is_citation_or_reference_like_parenthetical(content, language_code)
        )

    @staticmethod
    def is_citation_or_reference_like_parenthetical(content: str, language_code: str) -> bool:
        if language_code.lower() != "en":
            return False

        normalized = " ".join(content.split())
        if not normalized:
            return False

        reference_prefix = re.compile(
            r"^(?:see|cf\.?|compare|fig(?:ure)?s?\.?|ch(?:apter)?\.?|"
            r"p(?:age)?s?\.?|pp\.?|sec(?:tion)?s?\.?|vol(?:ume)?\.?|"
            r"no\.?|doi|isbn|§)\s",
            re.IGNORECASE,
        )
        if reference_prefix.search(normalized):
            return True

        # Numeric footnotes, page/range lists, and Roman-numeral references.
        if re.fullmatch(r"(?:\d+|[ivxlcdm]+)(?:\s*[-–,;]\s*(?:\d+|[ivxlcdm]+))*", normalized, re.IGNORECASE):
            return True

        # Common author-year citations, including "Smith et al., 2020".
        author = r"[A-Z][^\W\d_]*(?:['’\-][^\W\d_]+)?"
        author_list = rf"{author}(?:\s+(?:et\s+al\.?|and\s+{author}|&\s*{author}))*"
        author_year = rf"{author_list},?\s+(?:1[5-9]\d{{2}}|20\d{{2}})[a-z]?"
        return re.fullmatch(rf"{author_year}(?:\s*;\s*{author_year})*", normalized) is not None

    @staticmethod
    def ellipsis_phrase_break_offsets(sentence: str) -> list[int]:
        """Return phrase boundaries immediately after qualifying ellipses.

        An ellipsis is a maximal run of at least three consecutive dots, one
        or more Unicode ellipsis characters, or at least three dots separated
        by one or more literal spaces. A boundary is useful only when the text
        on each side contains vocalizable content. Tabs, line breaks, and other
        whitespace characters do not form a spaced ellipsis.
        """
        ellipsis_pattern = re.compile(r"\.{3,}|…+|\.(?: +\.){2,}")
        return [
            match.end()
            for match in ellipsis_pattern.finditer(sentence)
            if (
                app_text.is_vocalizable(sentence[:match.start()])
                and app_text.is_vocalizable(sentence[match.end():])
            )
        ]

    @staticmethod
    def double_dash_break_offsets(sentence: str) -> list[int]:
        """
        Returns split offsets (immediately after the second dash) for each
        double normal dash `--` that acts as a phrase break.

        A double dash is a break when it is a run of exactly two normal dashes,
        with optional whitespace on the left and/or on the right, and with
        vocalizable ("content") characters bounding the whole sequence:
            content \\s* -- \\s* content
        eg: "Hello--what are you doing?", "Hello -- what are you doing?"

        Single dashes, longer dash runs, and dashes bounded by punctuation
        or string edges are not breaks.
        """
        offsets: list[int] = []
        length = len(sentence)
        i = 0
        while i < length:
            if sentence[i] != "-":
                i += 1
                continue
            j = i
            while j < length and sentence[j] == "-":
                j += 1
            if j - i == 2:
                def has_content_before(index: int) -> bool:
                    k = index
                    while k >= 0 and sentence[k] in string.whitespace:
                        k -= 1
                    return k >= 0 and app_text.is_vocalizable(sentence[k])

                def has_content_after(index: int) -> bool:
                    k = index
                    while k < length and sentence[k] in string.whitespace:
                        k += 1
                    return k < length and app_text.is_vocalizable(sentence[k])

                if has_content_before(i - 1) and has_content_after(j):
                    offsets.append(j)
            i = j
        return offsets

    @staticmethod
    def long_phrase_to_phrases(phrase: Phrase, max_words:int) -> list[Phrase]:
        """
        Splits a phrase at arbitrary word if it exceeds max_words
        """
        num_phrases = math.ceil(phrase.num_words / max_words)
        stride = math.ceil(phrase.num_words / num_phrases)

        result = []
        for i in range(0, num_phrases):

            # TODO: if not at last phrase, if end of phrase is from a set of certain common words, move split point back 1 (eg, common particles, prepositions, pronouns, conjunction words...)

            start = (i + 0) * stride
            end   = (i + 1) * stride
            words = phrase.words[start:end]
            reason = phrase.reason if (i == num_phrases - 1) else Reason.WORD
            new_phrase = Phrase("".join(words), reason)
            result.append(new_phrase)
        return result

    @staticmethod
    def merge_ornamental_lines(phrases: list[Phrase]) -> list[Phrase]:
        """
        When a Phrase ends with PARAGRAPH or SPACE_BREAK and is 'not vocalizable', merges it with previous phrase
        Idea here is that these items must be typographical ornamentation lines that signify a section break
        """

        def is_ornamental_break(phrase: Phrase) -> bool:
            return phrase.reason in [Reason.PARAGRAPH, Reason.SPACE_BREAK] and not app_text.is_vocalizable(phrase.text)

        results: list[Phrase] = []
        leading_ornamental_phrases: list[Phrase] = []

        for phrase in phrases:
            if not is_ornamental_break(phrase):
                results.append(phrase)
            elif not results:
                # A leading ornamental line has no preceding phrase to merge
                # into. Hold it and prepend it to the first content phrase
                # below, so that the very first phrase of the text (and
                # therefore the first PhraseGroup of a section) starts with
                # vocalizable text instead of standing ornament-only.
                leading_ornamental_phrases.append(phrase)
            else:
                # Merge the non-vocalizable separator into the preceding phrase
                # and retain its structural meaning. Ornamental lines commonly
                # have only two trailing line feeds, so their initially inferred
                # reason is PARAGRAPH even though the ornament denotes a space
                # break between scenes.
                results[-1].text += phrase.text
                results[-1].reason = Reason.SPACE_BREAK

        if leading_ornamental_phrases:
            if results:
                # Attach leading ornaments to the first content phrase. The
                # ornament's own trailing break carries no boundary meaning at
                # the start of the text, so the content phrase keeps its reason.
                ornament_text = "".join(phrase.text for phrase in leading_ornamental_phrases)
                results[0].text = ornament_text + results[0].text
            else:
                # The text has no vocalizable content at all; leave as-is.
                results = leading_ornamental_phrases

        return results

    @staticmethod
    def downgrade_consecutive_space_breaks(phrases: list[Phrase]) -> list[Phrase]:
        """
        Downgrades immediate repeated SPACE_BREAK reasons to PARAGRAPH.

        Some EPUB-to-text converters emit multiple blank lines around adjacent headings,
        e.g. chapter number followed by chapter title. The first such break can be useful
        as a section/prosody marker, but repeated immediate SPACE_BREAK reasons overstate the
        structure and can trigger repeated section effects in downstream audio/browser flows.
        """

        last_reason_was_section = False

        for phrase in phrases:
            if phrase.reason == Reason.SPACE_BREAK:
                if last_reason_was_section:
                    phrase.reason = Reason.PARAGRAPH
                    phrase.text = phrase.text.rstrip() + "\n\n"
                last_reason_was_section = True
            else:
                last_reason_was_section = False

        return phrases
