from __future__ import annotations
from enum import Enum
from functools import total_ordering

from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *


class Phrase:
    def __init__(self, text: str, reason: Reason):
        self._text = text
        self.reason = reason
        self._words = TextUtil.get_words(self._text)

    @property
    def text(self) -> str:
        """ Read-only """
        return self._text

    @property
    def words(self) -> list[str]:
        return self._words

    @property
    def num_words(self) -> int:
        return len(self._words)

    def __str__(self):
        s = f"[Phrase] {str(self.reason.json_value).ljust(4)} text: {self.text.strip()}"
        return s

    def to_json_dict(self) -> dict:
        return {
            "text": self.text,
            "reason": self.reason.json_value # Not currently used by reader
        }

    @staticmethod
    def phrases_from_json_dicts(list_of_dicts: list[dict]) -> list[Phrase] | str:

        if not list_of_dicts or not isinstance(list_of_dicts, list):
            return f"bad type: {list_of_dicts}"

        result = []

        for item in list_of_dicts:

            if not isinstance(item, dict):
                return f"bad type: {item}"
            if not "text" in item:
                return f"missing required property: {item}"

            reason = Reason.from_json_value(item.get("reason"))
            phrase = Phrase(text=item["text"], reason=reason)
            result.append(phrase)

        return result

@total_ordering 
class Reason(tuple[int, str, float], Enum):
    """
    Describes the "reason" why a piece of text has been segmented.
    
    Value comparisons can be made between members directly or using the level property
    (eg, "reason1 < reason2" or "reason1.level < reason2.level")
    """
    
    # for back-compat and as fallback
    UNDEFINED = 0, "undefined", PAUSE_DURATION_SENTENCE
    # string has been split after an arbitrary word
    WORD = 1, "w", PAUSE_DURATION_WORD
    # string has been split at a phrase (not to be confused with the class named "Phrase")
    PHRASE = 2, "is", PAUSE_DURATION_PHRASE
    # string has been split at a sentence (which does not end in a paragraph break)
    SENTENCE = 3, "s", PAUSE_DURATION_SENTENCE
    # string has been split at a paragraph break (line feed)
    PARAGRAPH = 4, "p", PAUSE_DURATION_PARAGRAPH
    # string has been split at a paragraph break plus extra line feed/s
    SECTION = 5, "x", PAUSE_DURATION_SECTION

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value[0] < other.value[0]
        return NotImplemented
    
    @property
    def level(self) -> int:
        return self.value[0]

    @property
    def json_value(self) -> str:
        return self.value[1]

    @property
    def pause_duration(self) -> float:
        """
        Duration of silence that should be inserted
        between the given text segment and the one preceding it
        """
        return self.value[2]

    @classmethod
    def from_json_value(cls, value: str | None) -> Reason:
        if value is None:
            return Reason.UNDEFINED
        for member in cls:
            if member.json_value == value:
                return member
        return Reason.UNDEFINED

class PhraseGroup:
    """
    Wraps a list of `Phrase` instances.
    Gets passed to TTS inferencing routine.
    
    Reason for doing this rather than simply using a Phrase or simple string is that after TTS+STT, 
    we eventually want to forced-align the transcript with the individual phrases,
    ie, add 'phrase-level granularity' to the timing metadata.
    """
    
    def __init__(self, segments: list[Phrase] | None = None):
        self.phrases: list[Phrase] = segments or []

    @property
    def num_words(self) -> int:
        count = 0
        for phrase in self.phrases:
            count += phrase.num_words
        return count

    @property
    def text(self) -> str:
        """ Concatenated phrases text, preserving whitespace"""
        s = ""
        for phrase in self.phrases:
            s += phrase.text
        return s

    @property
    def presentable_text(self) -> str:
        """ Text in 'presentable' format for UI-related purposes """
        strings = [item.text.strip() for item in self.phrases]
        return " ".join(strings)
    
    @property
    def last_reason(self) -> Reason:
        return self.phrases[-1].reason if self.phrases else Reason.UNDEFINED

    @staticmethod
    def flatten_groups(groups: list[PhraseGroup]) -> list[Phrase]:
        phrases = []
        for group in groups:
            if group:
                phrases.extend(group.phrases)
        return phrases

    def as_flattened_phrase(self) -> Phrase:
        reason = self.phrases[-1].reason if self.phrases else Reason.UNDEFINED
        return Phrase(self.text, reason)

    def to_json_dict_list(self) -> list[dict]:
        return [phrase.to_json_dict() for phrase in self.phrases]

    @staticmethod
    def phrase_groups_from_json_list(list_of_lists: list[list[dict]]) -> list[PhraseGroup] | str:
        if not isinstance(list_of_lists, list):
            return f"Expected list of lists. Value: {str(list_of_lists)}"

        text_groups = []

        for lst in list_of_lists:

            if not isinstance(lst, list):
                return f"Expected list: {lst}"

            result = Phrase.phrases_from_json_dicts(lst)
            if isinstance(result, str):
                err = result
                return err

            phrases = result
            group = PhraseGroup(phrases)
            text_groups.append(group)

        return text_groups

    @staticmethod
    def phrase_groups_to_json_list(groups: list[PhraseGroup]) -> list[list[dict]]:
        if not groups:
            return []
        result = []
        for group in groups:
            if group:
                inner_list = group.to_json_dict_list()
                result.append(inner_list)
        return result

    @staticmethod
    def get_max_num_words(groups: list[PhraseGroup]) -> int:
        value = 0
        for group in groups:
            for phrase in group.phrases:
                value = max(value, phrase.num_words)
        return value
