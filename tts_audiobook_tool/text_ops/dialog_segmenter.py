from __future__ import annotations

from bisect import bisect_left, bisect_right
import string
import unicodedata

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason


_STRAIGHT_QUOTE = '"'
_CURLY_OPEN_QUOTE = "“"
_CURLY_CLOSE_QUOTE = "”"
_QUOTE_CHARS = {_STRAIGHT_QUOTE, _CURLY_OPEN_QUOTE, _CURLY_CLOSE_QUOTE}

# PhraseGroup voice indices are zero-based; this is voice sample 2 in the UI.
DIALOG_VOICE_INDEX = 1

# English attribution verbs that may follow a close-quote + speaker name
# (eg, "Some dialog," John said.). Matched case-insensitively, only when the
# language code is "en", as a STARTS-WITH test: the verb word in the text
# hits when it starts with one of these stems (an inflection of it), so a
# single stem per regular paradigm suffices — eg, "add" also covers "adds",
# "added", "adding". Irregular forms that do not start with the base stem
# ("said", "told", "cries", "replied", ...) are listed on their own.
_ATTRIBUTION_VERBS_EN = {
    # core
    "say", "said",
    "ask",
    "tell", "told",
    "reply", "replies", "replied",
    "add",
    "answer",
    "explain",
    "continue",
    # vocalization
    "exclaim",
    "shout",
    "whisper",
    "murmur",
    "mutter",
    "cry", "cries", "cried",
    "call",
    # attitude / argument
    "remark",
    "state",
    "note",
    "declare",
    "retort",
    "respond",
    "insist",
    "warn",
    "promise",
    "suggest",
    "repeat",
}


class DialogSegmenter:
    """Detect and segment dialog within existing phrase groups."""

    @staticmethod
    def segment_groups(
            groups: list[PhraseGroup],
            dialog_voice_index: int | None = None,
            language_code: str | None = None,
    ) -> list[PhraseGroup]:
        """
        Segment groups at accepted dialog boundaries without ever merging groups
        produced by the normal segmentation pass. A piece ending at a dialog
        span end gets reason PHRASE_QUOTE_END (ie, almost no pause before an
        attribution) when its continuation (a) starts with a lowercase
        alphabetic (eg, "Hello," she said.) or (b) when language_code is
        "en", starts with a capital-initial word whose next word is a
        whitelisted attribution verb (eg, "Some dialog," John said.). When
        dialog_voice_index is provided, assign it to every resulting group
        inside detected dialog.
        """
        if not groups:
            return []

        text = "".join(group.text for group in groups)
        group_ends: list[int] = []
        running_length = 0
        for group in groups:
            running_length += len(group.text)
            group_ends.append(running_length)

        dialog_spans = DialogSegmenter._find_dialog_spans(
            text,
            group_ends,
        )
        if not dialog_spans:
            return groups

        span_end_offsets = {end for _, end in dialog_spans}
        boundaries = sorted({
            boundary
            for span in dialog_spans
            for boundary in span
        })
        result: list[PhraseGroup] = []
        group_start = 0
        for index, group in enumerate(groups):
            group_end = group_start + len(group.text)
            first_boundary = bisect_right(boundaries, group_start)
            last_boundary = bisect_left(boundaries, group_end)
            local_boundaries = [
                boundary - group_start
                for boundary in boundaries[first_boundary:last_boundary]
            ]
            parts = DialogSegmenter._split_group(
                group,
                local_boundaries,
                group_start,
                span_end_offsets,
                text,
                language_code,
            )
            # A dialog span ending exactly at this group's boundary still
            # continues almost without a pause when the next group (if any)
            # starts with an attribution (lowercase alphabetic, or for
            # "en" a speaker name followed by a whitelisted verb).
            if (
                group_end in span_end_offsets
                and DialogSegmenter._continues_with_attribution(
                    text,
                    group_end,
                    language_code,
                )
            ):
                parts[-1] = DialogSegmenter._with_quote_end_reason(parts[-1])
            result.extend(parts)
            group_start = group_end

        if dialog_voice_index is None:
            return result
        return DialogSegmenter._assign_dialog_voice(
            result,
            dialog_spans,
            dialog_voice_index,
        )

    @staticmethod
    def _find_dialog_spans(
            text: str,
            group_ends: list[int],
    ) -> list[tuple[int, int]]:
        quote_pairs: list[tuple[int, int]] = []
        paragraph_start = 0

        # A physical line is a paragraph boundary for quote-pairing purposes.
        for line in text.splitlines(keepends=True):
            content_length = len(line.rstrip("\r\n"))
            paragraph = line[:content_length]
            quote_pairs.extend(
                DialogSegmenter._find_quote_pairs_in_paragraph(
                    paragraph,
                    paragraph_start,
                )
            )
            paragraph_start += len(line)

        if paragraph_start < len(text):
            quote_pairs.extend(
                DialogSegmenter._find_quote_pairs_in_paragraph(
                    text[paragraph_start:],
                    paragraph_start,
                )
            )

        dialog_pairs = [
            pair
            for pair in quote_pairs
            if DialogSegmenter._is_dialog_pair(
                text,
                *pair,
                group_ends,
            )
        ]

        # Overlapping pairs can only result from nested/interleaved double-quote
        # styles. Double-quote nesting is not interpreted specially; retain the
        # outer, left-most accepted pair.
        non_overlapping_dialog_pairs: list[tuple[int, int]] = []
        for pair in sorted(dialog_pairs, key=lambda item: (item[0], -item[1])):
            if (
                non_overlapping_dialog_pairs
                and pair[0] < non_overlapping_dialog_pairs[-1][1]
            ):
                continue
            non_overlapping_dialog_pairs.append(pair)

        dialog_spans: list[tuple[int, int]] = []
        for opening_quote_index, closing_quote_index in non_overlapping_dialog_pairs:
            containing_group_end = group_ends[
                bisect_right(group_ends, closing_quote_index)
            ]
            dialog_spans.append((
                opening_quote_index,
                DialogSegmenter._end_after_attached_punctuation(
                    text,
                    closing_quote_index,
                    containing_group_end,
                ),
            ))
        return dialog_spans

    @staticmethod
    def _find_quote_pairs_in_paragraph(
        paragraph: str,
        offset: int,
    ) -> list[tuple[int, int]]:
        quote_pairs: list[tuple[int, int]] = []
        opening_quote_indices: list[int] = []

        # Context is needed for straight quotes and also lets imperfectly
        # normalized source text use mixed curly/straight pairs. A stack allows
        # a valid inner pair to survive an earlier unmatched opening quote.
        for index, char in enumerate(paragraph):
            if char not in _QUOTE_CHARS:
                continue

            looks_closing = (
                char == _CURLY_CLOSE_QUOTE
                or DialogSegmenter._looks_like_closing_quote(
                    paragraph,
                    index,
                )
            )
            if opening_quote_indices and looks_closing:
                opening_quote_index = opening_quote_indices.pop()
                quote_pairs.append((offset + opening_quote_index, offset + index))
                continue

            looks_opening = (
                char == _CURLY_OPEN_QUOTE
                or DialogSegmenter._looks_like_opening_quote(
                    paragraph,
                    index,
                )
            )
            if looks_opening:
                opening_quote_indices.append(index)

        return quote_pairs

    @staticmethod
    def _looks_like_opening_quote(text: str, index: int) -> bool:
        next_index = DialogSegmenter._next_non_whitespace_index(
            text,
            index + 1,
        )
        if next_index is None:
            return False
        if index == 0 or text[index - 1].isspace():
            return True

        previous_char = text[index - 1]
        return previous_char in "([{<:;,—–-"

    @staticmethod
    def _looks_like_closing_quote(text: str, index: int) -> bool:
        previous_index = DialogSegmenter._previous_non_whitespace_index(
            text,
            index - 1,
        )
        if previous_index is None:
            return False
        if index == len(text) - 1 or text[index + 1].isspace():
            return True

        next_char = text[index + 1]
        return unicodedata.category(next_char).startswith("P")

    @staticmethod
    def _next_non_whitespace_index(text: str, start: int) -> int | None:
        for index in range(start, len(text)):
            if not text[index].isspace():
                return index
        return None

    @staticmethod
    def _previous_non_whitespace_index(text: str, start: int) -> int | None:
        for index in range(start, -1, -1):
            if not text[index].isspace():
                return index
        return None

    @staticmethod
    def _is_dialog_pair(
        text: str,
        opening_quote_index: int,
        closing_quote_index: int,
        group_ends: list[int],
    ) -> bool:
        content = text[opening_quote_index + 1:closing_quote_index]
        if not app_text.is_vocalizable(content):
            return False

        for char in content:
            if unicodedata.category(char).startswith("L"):
                if char.isupper():
                    return True
                break
            if unicodedata.category(char).startswith("N"):
                break

        # Lowercase, numeric, and caseless openings are accepted when structural
        # evidence makes dialog more likely. Otherwise, retain the conservative
        # behavior for short inline labels, emphasized terms, and scare quotes.
        if app_text.get_word_count(content, vocalizable_only=True) > 3:
            return True

        opening_group_index = bisect_right(group_ends, opening_quote_index)
        closing_group_index = bisect_right(group_ends, closing_quote_index)
        if opening_group_index != closing_group_index:
            return True

        paragraph_start = max(
            text.rfind("\n", 0, opening_quote_index),
            text.rfind("\r", 0, opening_quote_index),
        ) + 1
        text_before_opening = text[paragraph_start:opening_quote_index]
        if not text_before_opening.strip():
            return True

        previous_text = text_before_opening.rstrip()
        if previous_text and previous_text[-1] in {",", ":", "—", "–"}:
            return True

        stripped_content = content.lstrip()
        if stripped_content.startswith(("—", "–")):
            return True

        return "?" in content or "!" in content

    @staticmethod
    def _end_after_attached_punctuation(
        text: str,
        closing_quote_index: int,
        containing_group_end: int,
    ) -> int:
        index = closing_quote_index + 1

        # Punctuation immediately after the closing quote belongs to the quote.
        # Do not cross an existing group boundary to collect it: this pass may
        # subdivide first-pass groups, but must never recombine them.
        while (
            index < containing_group_end
            and text[index] not in _QUOTE_CHARS
            and unicodedata.category(text[index]).startswith("P")
        ):
            index += 1

        # Preserve the project's convention that boundary whitespace belongs to
        # the preceding segment. This also avoids punctuation/line-break-only
        # groups at paragraph ends.
        while (
            index < containing_group_end
            and text[index] in string.whitespace
        ):
            index += 1

        return index

    @staticmethod
    def _assign_dialog_voice(
            groups: list[PhraseGroup],
            dialog_spans: list[tuple[int, int]],
            dialog_voice_index: int,
    ) -> list[PhraseGroup]:
        """Assign dialog groups without mutating source groups that were reused."""
        result: list[PhraseGroup] = []
        span_index = 0
        group_start = 0

        for group in groups:
            group_end = group_start + len(group.text)
            while (
                span_index < len(dialog_spans)
                and dialog_spans[span_index][1] <= group_start
            ):
                span_index += 1

            is_dialog = (
                span_index < len(dialog_spans)
                and dialog_spans[span_index][0] <= group_start
                and group_end <= dialog_spans[span_index][1]
            )
            if is_dialog and group.voice_index != dialog_voice_index:
                result.append(PhraseGroup(
                    list(group.phrases),
                    voice_index=dialog_voice_index,
                ))
            else:
                result.append(group)
            group_start = group_end

        return result

    @staticmethod
    def _split_group(
        group: PhraseGroup,
        local_boundaries: list[int],
        group_start: int,
        span_end_offsets: set[int],
        text: str,
        language_code: str | None,
    ) -> list[PhraseGroup]:
        """
        Split `group` at the given local (group-relative) dialog boundaries.

        A piece ending at a dialog span end gets reason PHRASE_QUOTE_END when
        its continuation is an attribution (lowercase alphabetic, or for
        "en" a speaker name followed by a whitelisted verb); other mid-phrase
        cuts keep reason PHRASE.
        """
        if not local_boundaries:
            return [group]

        boundaries = sorted(set(local_boundaries))
        result: list[PhraseGroup] = []
        current_phrases: list[Phrase] = []
        group_position = 0
        boundary_index = 0

        for phrase in group.phrases:
            phrase_start = group_position
            phrase_end = phrase_start + len(phrase.text)
            piece_start = 0

            while (
                boundary_index < len(boundaries)
                and boundaries[boundary_index] <= phrase_end
            ):
                boundary = boundaries[boundary_index]
                piece_end = boundary - phrase_start
                piece_text = phrase.text[piece_start:piece_end]
                if piece_text:
                    absolute_boundary = group_start + boundary
                    if (
                        absolute_boundary in span_end_offsets
                        and DialogSegmenter._continues_with_attribution(
                            text,
                            absolute_boundary,
                            language_code,
                        )
                    ):
                        reason = Reason.PHRASE_QUOTE_END
                    elif piece_end == len(phrase.text):
                        reason = phrase.reason
                    else:
                        reason = Reason.PHRASE
                    current_phrases.append(Phrase(piece_text, reason))

                if current_phrases:
                    result.append(
                        PhraseGroup(
                            current_phrases,
                            voice_index=group.voice_index,
                        )
                    )
                    current_phrases = []

                piece_start = piece_end
                boundary_index += 1

            remainder = phrase.text[piece_start:]
            if remainder:
                current_phrases.append(Phrase(remainder, phrase.reason))

            group_position = phrase_end

        if current_phrases:
            result.append(PhraseGroup(current_phrases, voice_index=group.voice_index))

        return result

    @staticmethod
    def _continues_with_lowercase_alpha(text: str, offset: int) -> bool:
        """
        Whether the first non-whitespace character at or after `offset` in
        `text` is a lowercase alphabetic. Whitespace is normally consumed by
        the preceding segment (see _end_after_attached_punctuation), so this
        is usually the character right at `offset`; the skip is defensive for
        boundaries that coincide with group ends.
        """
        index = offset
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return False
        char = text[index]
        return char.isalpha() and char.islower()

    @staticmethod
    def _continues_with_attribution(
        text: str,
        offset: int,
        language_code: str | None,
    ) -> bool:
        """
        Whether the text at or after `offset` continues with an attribution
        warranting an almost no pause: a lowercase alphabetic continuation
        (any language) or, when the language code is "en", a capital-initial
        word whose next word is a whitelisted attribution verb (eg,
        "Some dialog," John said.).
        """
        if DialogSegmenter._continues_with_lowercase_alpha(text, offset):
            return True
        if (language_code or "") != "en":
            return False
        return DialogSegmenter._continues_with_name_verb(text, offset)

    @staticmethod
    def _continues_with_name_verb(text: str, offset: int) -> bool:
        """
        Whether the text at or after `offset` begins with a capital-initial
        word (a speaker name) whose next word starts with one of
        _ATTRIBUTION_VERBS_EN (case-insensitively), so inflected forms are
        recognized — eg, stem "say" also matches "says", "said" is its own
        stem, and "add" matches "adding". Leading whitespace is skipped (as
        in _continues_with_lowercase_alpha), but only spaces/tabs may
        separate the name and the verb: a line break between them defeats
        the match.
        """
        index = offset
        length = len(text)

        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            return False

        first_char = text[index]
        if not (first_char.isalpha() and first_char.isupper()):
            return False

        index += 1
        while index < length and not text[index].isspace():
            index += 1

        while index < length and text[index] in " \t":
            index += 1
        if index >= length or not text[index].isalpha():
            return False

        verb_start = index
        while index < length and text[index].isalpha():
            index += 1

        verb = text[verb_start:index].lower()
        return any(verb.startswith(stem) for stem in _ATTRIBUTION_VERBS_EN)

    @staticmethod
    def _with_quote_end_reason(group: PhraseGroup) -> PhraseGroup:
        """
        Return a new group whose final phrase has reason PHRASE_QUOTE_END.
        Never mutates the source group, which may be shared with the caller.
        """
        if not group.phrases:
            return group
        phrases = list(group.phrases[:-1])
        phrases.append(Phrase(group.phrases[-1].text, Reason.PHRASE_QUOTE_END))
        return PhraseGroup(phrases, voice_index=group.voice_index)
