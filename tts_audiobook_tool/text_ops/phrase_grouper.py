import re
from collections.abc import Collection

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.constants_config import SHORT_GROUP_MERGE_MAX_WORDS
from tts_audiobook_tool.text_ops.phrase_segmenter import Reason, Phrase, PhraseSegmenter
from tts_audiobook_tool.text_ops.dialog_segmenter import (
    DIALOG_VOICE_INDEX,
    DialogSegmenter,
)
from tts_audiobook_tool.util import printt

class PhraseGrouper:
    """
    Creates PhraseGroups from Phrases or strings
    """

    @staticmethod
    def text_to_groups(
            text: str,
            max_words: int,
            strategy: SegmentationStrategy=SegmentationStrategy.SENTENCE_PLUS,
            pysbd_lang: str="en",
            dialog_segmentation: bool=False,
            heading_texts: Collection[str] | None=None,
            merge_short_sentences: bool=False,
    ) -> list[PhraseGroup]:
        """
        Creates PhraseGroups using the passed-in raw source text.
        
        This is the app's main, high-level function for chunking text.
        One PhraseGroup gets transformed into one TTS prompt.

        When dialog_segmentation is enabled, a final pass subdivides groups so
        accepted dialog and narration do not share a group, preassigns dialog
        to voice sample 2, and marks close-quote pieces followed by an
        attribution with reason PHRASE_QUOTE_END (a lowercase continuation, or
        for language code "en" a speaker name followed by a whitelisted verb).
        It does not recombine groups created by the normal segmentation passes.

        heading_texts are the texts of paragraphs which are known to be headings
        (eg, from EPUB markup). Those paragraphs end with reason HEADING.

        When merge_short_sentences is enabled, a final pass merges short groups
        into a neighboring group, across paragraph boundaries (see
        merge_short_groups_across_paragraphs). Dialog segmentation is skipped
        in that case.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # This guarantees that imported text has no blank lines containing
        # whitespace, and that no other line ends with spaces or tabs.
        text = re.sub(r"[ \t]+\n", "\n", text)

        phrases = PhraseSegmenter.text_to_phrases(text, max_words=max_words, pysbd_lang=pysbd_lang)
        if heading_texts:
            PhraseGrouper.mark_heading_paragraphs(phrases, heading_texts)

        # First group by either complete sentence or paragraph
        match strategy:
            case SegmentationStrategy.SENTENCE:
                reason_threshold = Reason.SENTENCE
            case SegmentationStrategy.SENTENCE_PLUS:
                reason_threshold = Reason.SENTENCE
            case SegmentationStrategy.MULTI_SENTENCE:
                reason_threshold = Reason.SENTENCE # (will re-combine sentences later)
            case SegmentationStrategy.MAX_LEN:
                reason_threshold = Reason.PARAGRAPH
        groups = PhraseGrouper.phrases_to_groups_by_reason(phrases, reason_threshold)

        if strategy == SegmentationStrategy.SENTENCE_PLUS:
            # Mitigates tts glitches due to too-short prompts
            SHORT_SENTENCE_NUM_WORDS = 2
            groups = PhraseGrouper.merge_short_sentences(groups, SHORT_SENTENCE_NUM_WORDS, max_words)
        elif strategy == SegmentationStrategy.MULTI_SENTENCE:
            # Piggybacks off same logic
            groups = PhraseGrouper.merge_short_sentences(groups, 9999, max_words)

        # Split group when exceeds max_words
        results = []
        for group in groups:
            split_groups = PhraseGrouper.group_to_groups_by_max_words(group, max_words)
            results.extend(split_groups)
        groups = results

        if merge_short_sentences:
            groups = PhraseGrouper.merge_short_groups_across_paragraphs(
                groups, SHORT_GROUP_MERGE_MAX_WORDS, max_words
            )
        elif dialog_segmentation:
            groups = DialogSegmenter.segment_groups(
                groups,
                dialog_voice_index=DIALOG_VOICE_INDEX,
                language_code=pysbd_lang,
            )

        groups = PhraseGrouper.merge_ornamental_groups(groups)

        return groups

    @staticmethod
    def mark_heading_paragraphs(phrases: list[Phrase], heading_texts: Collection[str]) -> None:
        """
        Changes the reason of each paragraph-ending phrase from PARAGRAPH to
        HEADING when the paragraph's text is one of heading_texts
        (compared with normalized whitespace).
        """
        normalized_headings = {normalize_heading_text(text) for text in heading_texts}
        normalized_headings.discard("")
        paragraph_text = ""
        for phrase in phrases:
            paragraph_text += phrase.text
            if phrase.reason < Reason.PARAGRAPH:
                continue
            if phrase.reason == Reason.PARAGRAPH and \
                    normalize_heading_text(paragraph_text) in normalized_headings:
                phrase.reason = Reason.HEADING
            paragraph_text = ""

    @staticmethod
    def merge_ornamental_groups(groups: list[PhraseGroup]) -> list[PhraseGroup]:
        """
        Folds ornamental-only groups into a vocalizable neighboring group.

        A PhraseGroup is the app's atomic TTS prompt unit, and a group whose
        text has no vocalizable content (a lone dinkus or divider line)
        normalizes to an empty prompt. Segmentation already merges ornamental
        lines into a neighboring phrase, but that invariant can be undone
        later: an ornament trailing the text has no trailing line breaks (so
        its reason stays SENTENCE and it never merges backward), and the
        optional dialog pass re-splits a merged phrase at quote boundaries,
        isolating its ornament again. This final pass restores the invariant
        for every position: mid-text and trailing ornamental-only groups
        merge backward into the previous group, and leading ones merge
        forward into the first vocalizable group.

        Groups are returned unchanged when no group has vocalizable text.
        """

        result: list[PhraseGroup] = []
        leading: list[PhraseGroup] = []

        for group in groups:
            if app_text.is_vocalizable(group.text):
                result.append(group)
                continue
            if not result:
                leading.append(group)
                continue
            # Merge backward, mirroring PhraseSegmenter.merge_ornamental_lines:
            # the ornament joins the previous group's last phrase and promotes
            # its boundary to a space break.
            if result[-1].phrases:
                last_phrase = result[-1].phrases[-1]
                last_phrase.text += group.text
                last_phrase.reason = Reason.SPACE_BREAK
                result[-1].invalidate_presentable_memos()
            else:
                result[-1].phrases.extend(group.phrases)
                result[-1].invalidate_presentable_memos()

        if leading:
            if result:
                # Attach leading ornaments to the first content phrase. The
                # ornament's own break carries no boundary meaning at the
                # start of the text, so the content phrase keeps its reason.
                ornament_text = "".join(group.text for group in leading)
                result[0].phrases[0].text = ornament_text + result[0].phrases[0].text
                result[0].invalidate_presentable_memos()
            else:
                # The text has no vocalizable content at all; leave as-is.
                result = leading

        return result

    @staticmethod
    def group_to_groups_by_max_words(group: PhraseGroup, max_words: int) -> list[PhraseGroup]:
        """
        Breaks up a PhraseGroup into multiple groups if its num_words > max_words.
        A phrase which already exceeds max_words remains in its own non-empty group.
        """
        result = []
        temp_group = PhraseGroup()

        for phrase in group.phrases:

            if temp_group.phrases and temp_group.num_words + phrase.num_words > max_words:
                result.append(temp_group)
                temp_group = PhraseGroup()

            temp_group.phrases.append(phrase)

        if temp_group.phrases:
            result.append(temp_group)

        # TODO: second pass, attempt to balance the num_words of each group maybe
        # eg, 10/10/10, 10, 10 -> 10/10, 10/10, 10/10

        return result

    @staticmethod
    def phrases_to_groups_by_reason(phrases: list[Phrase], reason_threshold: Reason) -> list[PhraseGroup]:
        """
        Groups phrases into PhraseGroups based on reason threshold
        (ie, when a phrase has the given reason's value or greater, it will cut a new group)
        """

        groups: list[PhraseGroup] = []
        group = PhraseGroup()

        for phrase in phrases:
            group.phrases.append(phrase)
            if phrase.reason.level >= reason_threshold.level:
                groups.append(group)
                group = PhraseGroup()
        if group:
            groups.append(group)

        return groups

    @staticmethod
    def merge_short_sentences(
            groups: list[PhraseGroup], 
            shortness_threshold: int,
            max_words: int
    ) -> list[PhraseGroup]:
        """
        Merges short single phrase group with neighbor if doesn't exceed max_words 
        and does not cross paragraph/section boundary. Uses two passes.

        Again, the reason for this operation is to prevent very short prompts,
        which TTS models do not like.
        """
        # Merge with previous
        new_groups = []
        for i, group in enumerate(groups):  
            did_merge = False            
            is_single_short_any = len(group.phrases) == 1 and group.num_words <= shortness_threshold
            if is_single_short_any and len(new_groups) > 0:                
                last_new_group = new_groups[-1]
                can_merge = last_new_group.num_words + group.num_words <= max_words and \
                    last_new_group.last_reason < Reason.PARAGRAPH
                if can_merge:
                    new_groups[-1].phrases.append(group.phrases[0])
                    new_groups[-1].invalidate_presentable_memos()
                    did_merge = True
            if not did_merge:
                new_groups.append(group)

        # Merge with next
        groups = new_groups
        new_groups: list[PhraseGroup] = []
        for i, group in enumerate(groups):            
            did_merge = False
            is_short_sentence = len(group.phrases) == 1 and \
                group.num_words <= shortness_threshold and \
                group.last_reason <= Reason.SENTENCE
            if is_short_sentence and i + 1 < len(groups):
                next_group = groups[i+1]
                can_merge = next_group.num_words + group.num_words <= max_words
                if can_merge:
                    next_group.phrases.insert(0, group.phrases[0])
                    next_group.invalidate_presentable_memos()
                    did_merge = True
            if not did_merge:
                new_groups.append(group)

        return new_groups            

    @staticmethod
    def merge_short_groups_across_paragraphs(
            groups: list[PhraseGroup],
            shortness_threshold: int,
            max_words: int
    ) -> list[PhraseGroup]:
        """
        Merges each group with shortness_threshold words or fewer into the
        previous group, or if that would exceed max_words, into the next group.
        Unlike merge_short_sentences, this crosses paragraph boundaries, which
        helps with texts that put every line into its own paragraph.
        Space breaks and section breaks are not crossed.

        Headings only merge with each other, so that eg "Chapter 1" and the
        chapter title get combined, but not the chapter title and the
        chapter's body text. Headings are heading-like groups (see
        is_heading_like), plus a title following a heading at the very start
        of the text, whatever its punctuation (eg "Chapter 14" followed by
        "What can we do?"). Such a title must be a short paragraph which does
        not start like dialog.

        Where a paragraph without punctuation gets joined with the next one,
        a period is added to it (see _punctuate_paragraph_join). Headings
        without punctuation at their end get a period there as well.
        """
        title_id: int | None = None
        if len(groups) >= 2 and PhraseGrouper.is_heading_like(groups[0]):
            title = groups[1]
            is_title = title.num_words <= shortness_threshold and \
                title.last_reason >= Reason.PARAGRAPH and \
                not title.text.lstrip().startswith(PhraseGrouper._DIALOG_START_CHARS)
            if is_title:
                title_id = id(title)

        def is_heading(group: PhraseGroup) -> bool:
            return id(group) == title_id or PhraseGrouper.is_heading_like(group)

        result: list[PhraseGroup] = []

        for i, group in enumerate(groups):
            if group.num_words > shortness_threshold:
                result.append(group)
                continue

            group_is_heading = is_heading(group)

            # Merge with previous
            if result:
                previous = result[-1]
                can_merge = previous.num_words + group.num_words <= max_words and \
                    previous.last_reason < Reason.SPACE_BREAK and \
                    is_heading(previous) == group_is_heading
                if can_merge:
                    PhraseGrouper._punctuate_paragraph_join(previous.phrases[-1])
                    previous.phrases.extend(group.phrases)
                    previous.invalidate_presentable_memos()
                    PhraseGrouper._promote_heading_end(previous)
                    continue

            # Merge with next (which then gets processed in turn)
            if i + 1 < len(groups):
                next_group = groups[i + 1]
                can_merge = next_group.num_words + group.num_words <= max_words and \
                    group.last_reason < Reason.SPACE_BREAK and \
                    is_heading(next_group) == group_is_heading
                if can_merge:
                    PhraseGrouper._punctuate_paragraph_join(group.phrases[-1])
                    next_group.phrases[0:0] = group.phrases
                    next_group.invalidate_presentable_memos()
                    PhraseGrouper._promote_heading_end(next_group)
                    continue

            result.append(group)

        # End headings with a period, too, which gives them a more natural intonation
        for group in result:
            if is_heading(group) and app_text.lacks_final_punctuation(group.text):
                group.phrases[-1].text = app_text.add_period_after_last_word(group.phrases[-1].text)
                group.invalidate_presentable_memos()

        return result

    @staticmethod
    def _punctuate_paragraph_join(phrase: Phrase) -> None:
        """
        Adds a period to a paragraph-ending phrase without punctuation which is
        about to be joined with the following paragraph (eg "Chapter 1" and the
        chapter title). Line breaks are removed from TTS prompts, so the model
        would otherwise read both lines as one phrase without a pause.
        """
        if phrase.reason >= Reason.PARAGRAPH and app_text.lacks_final_punctuation(phrase.text):
            phrase.text = app_text.add_period_after_last_word(phrase.text)

    @staticmethod
    def _promote_heading_end(group: PhraseGroup) -> None:
        """
        When headings were merged (eg "Chapter 1" with reason HEADING and the
        chapter title), the HEADING reason moves to the end of the merged group,
        so that the heading pause follows the title. Inner HEADING reasons
        become PARAGRAPH.
        """
        if group.last_reason != Reason.PARAGRAPH:
            return
        inner_headings = [phrase for phrase in group.phrases[:-1] if phrase.reason == Reason.HEADING]
        if not inner_headings:
            return
        for phrase in inner_headings:
            phrase.reason = Reason.PARAGRAPH
        group.phrases[-1].reason = Reason.HEADING

    # Characters that start a line of dialog
    _DIALOG_START_CHARS = ("\"", "“", "”", "„", "»", "«", "—", "–")

    @staticmethod
    def is_heading_like(group: PhraseGroup) -> bool:
        """
        Returns True if the group ends with reason HEADING, or if its text ends
        without sentence-ending punctuation, as headings do (eg "Chapter 1", a
        chapter title, or the lines of a title page). Trailing ornaments (eg a
        dinkus) are ignored.

        Groups that were split mid-sentence (reason below SENTENCE) are never
        heading-like. A group at the very end of the text has reason SENTENCE
        even without punctuation, so it is evaluated like a paragraph.
        """
        if group.last_reason == Reason.HEADING:
            return True
        if group.last_reason < Reason.SENTENCE:
            return False
        return app_text.ends_without_punctuation(group.text)

    @staticmethod
    def print_groups(groups: list[PhraseGroup]) -> None:
        """ For debugging """
        for group in groups:
            printt(f"Group: ({group.num_words} words)")
            for i, phrase in enumerate(group.phrases):
                s = f"{phrase.reason} (words: {phrase.num_words})"
                if phrase.reason >= Reason.PARAGRAPH:
                    s += " -----"
                printt(f"  {repr(phrase.text)} {s}")
            printt()


def normalize_heading_text(text: str) -> str:
    """ Normalizes text for comparing headings: collapsed whitespace, case-insensitive """
    return " ".join(text.split()).casefold()
