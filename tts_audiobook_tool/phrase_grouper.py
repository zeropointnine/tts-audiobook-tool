import math
from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.phrase_segmenter import Reason, Phrase, PhraseSegmenter
from tts_audiobook_tool.util import printt

class PhraseGrouper:
    """
    Creates PhraseGroups from Phrases or strings
    """

    @staticmethod
    def text_to_groups(
            text: str,
            max_words: int,
            strategy: SegmentationStrategy=SegmentationStrategy.NORMAL,
            pysbd_lang: str="en"
    ) -> list[PhraseGroup]:
        """
        Creates PhraseGroups using the passed-in raw source text.
        
        This is the app's main, high-level function for chunking text.
        One PhraseGroup gets transformed into one TTS prompt.
        """
        phrases = PhraseSegmenter.text_to_phrases(text, max_words=max_words, pysbd_lang=pysbd_lang)

        # First group by either complete sentence or paragraph
        match strategy:
            case SegmentationStrategy.NORMAL:
                reason_threshold = Reason.SENTENCE
            case SegmentationStrategy.MULTI_SENTENCE:
                reason_threshold = Reason.SENTENCE # (will re-combine sentences later)
            case SegmentationStrategy.MAX_LEN:
                reason_threshold = Reason.PARAGRAPH
        groups = PhraseGrouper.phrases_to_groups_by_reason(phrases, reason_threshold)

        if strategy == SegmentationStrategy.NORMAL:
            # Special case for NORMAL:
            # Mitigate tts glitches due to too-short prompts
            SHORT_SENTENCE_NUM_WORDS = 2
            groups = PhraseGrouper.merge_short_sentences(groups, SHORT_SENTENCE_NUM_WORDS, max_words)
        elif strategy == SegmentationStrategy.MULTI_SENTENCE:
            # Piggybacks off same logic
            groups = PhraseGrouper.merge_short_sentences(groups, 9999, max_words)

        # Split group when exceeds max_words
        results = []
        for group in groups:
            groups = PhraseGrouper.group_to_groups_by_max_words(group, max_words)
            results.extend(groups)
        groups = results

        return groups

    @staticmethod
    def group_to_groups_by_max_words(group: PhraseGroup, max_words: int) -> list[PhraseGroup]:
        """
        Breaks up a PhraseGroup into multiple groups if its num_words > max_words.
        Assumes all phrases' num_words <= max_words
        """
        result = []
        temp_group = PhraseGroup()

        for phrase in group.phrases:

            if temp_group.num_words + phrase.num_words > max_words:
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
                    did_merge = True
            if not did_merge:
                new_groups.append(group)

        return new_groups            

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
