from __future__ import annotations
from enum import Enum
from functools import total_ordering

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

""" Phrase, PhraseGroup, and Reason classes"""

class Phrase:
    def __init__(self, text: str, reason: Reason):
        self._text = text
        self.reason = reason
        self._words = app_text.get_words(self._text, vocalizable_only=False)

    def __eq__(self, other: Any):
        if not isinstance(other, Phrase):
            return NotImplemented
        return self._text == other._text and self.reason == other.reason
    
    def __str__(self):
        s = f"[Phrase] reason: {str(self.reason.json_value)}, text: {repr(self._text)}"
        return s

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value
        self._words = app_text.get_words(self._text, vocalizable_only=False)

    @property
    def presentable_text(self) -> str:
        """ Text in 'presentable' format for UI-related purposes """
        text = self.text.replace("\n", " ").replace("\r", " ")
        text = app_text.massage_post_normalize(text)
        return text

    @property
    def words(self) -> list[str]:
        """ Returns the words in the phrase (with all characters preserved) """
        return self._words

    @property
    def num_words(self) -> int:
        return len(self._words)

    def to_json_dict(self) -> dict:
        return {
            "text": self.text,
            "reason": self.reason.json_value # Not currently used by reader
        }

    @staticmethod
    def phrases_from_json_dicts(list_of_dicts: list[dict]) -> list[Phrase] | str:

        if not isinstance(list_of_dicts, list):
            return f"Expected phrase list. Value: {list_of_dicts}"
        if not list_of_dicts:
            return "Phrase list contains no phrases"

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

class PhraseGroup:
    """
    Wraps a list of `Phrase` instances.
    This is the app's atomic unit of TTS text inference.
    
    Reason for this higher-level wrapper is that after TTS+STT, 
    we can force-align the transcript with the PhraseGroup's constituent phrases.
    Ie, add 'phrase-level granularity' to the timing metadata.
    """
    
    def __init__(self, phrases: list[Phrase] | None = None, voice_index: int = -1):
        self.phrases: list[Phrase] = phrases or []
        self.voice_index = voice_index

    @property
    def voice_index(self) -> int:
        return self.voice_index_value

    @voice_index.setter
    def voice_index(self, value: int) -> None:
        self.voice_index_value = value

    @property
    def num_words(self) -> int:
        count = 0
        for phrase in self.phrases:
            count += phrase.num_words
        return count

    @property
    def text(self) -> str:
        """ 
        Concatenated phrases text. Does not modify whitespace."""
        s = ""
        for phrase in self.phrases:
            s += phrase.text
        return s

    @property
    def presentable_text(self) -> str:
        """ Text in 'presentable' format for UI-related purposes """
        text = ""
        for phrase in self.phrases:
            text += phrase.text + " "
        text = text.replace("\n", " ").replace("\r", " ")
        text = app_text.massage_post_normalize(text)
        return text
    
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

    def to_json_dict(self) -> dict:
        return {
            "voice_index": self.voice_index,
            "phrases": self.to_json_dict_list(),
        }

    @staticmethod
    def phrase_groups_from_json_list(values: list[Any]) -> list[PhraseGroup] | str:
        if not isinstance(values, list):
            return f"Expected list of phrase groups. Value: {str(values)}"

        text_groups = []

        for group_index, value in enumerate(values):
            if isinstance(value, dict):
                if "phrases" not in value:
                    return f"Phrase group missing 'phrases': {value}"
                phrase_dicts = value["phrases"]
                raw_voice_index = value.get("voice_index", -1)
                voice_index = (
                    raw_voice_index
                    if isinstance(raw_voice_index, int)
                    and not isinstance(raw_voice_index, bool)
                    and raw_voice_index >= -1
                    else -1
                )
            else:
                phrase_dicts = value
                voice_index = -1

            if not isinstance(phrase_dicts, list):
                return f"Expected phrase list: {phrase_dicts}"
            if not phrase_dicts:
                # INTERNAL:
                # From November 2025 through August 2026, the phrase grouper could
                # emit an empty group when its first phrase already exceeded max_words.
                # One trigger was post-split merging of ornamental/non-vocalizable
                # lines, which could grow a phrase beyond the limit after splitting.
                #
                # An empty phrase group is fatal because project line indices are tied to
                # generated audio and metadata. Do not drop, merge, or synthesize content
                # here; those remediations can silently alter index identity or book text.
                return f"Phrase group {group_index + 1} contains no phrases"

            result = Phrase.phrases_from_json_dicts(phrase_dicts)
            if isinstance(result, str):
                err = result
                return err

            phrases = result
            group = PhraseGroup(phrases, voice_index=voice_index)
            text_groups.append(group)

        return text_groups

    @staticmethod
    def phrase_groups_to_json_list(groups: list[PhraseGroup]) -> list[dict]:
        if not groups:
            return []
        result = []
        for group in groups:
            if group:
                result.append(group.to_json_dict())
        return result

    @staticmethod
    def get_max_num_words(groups: list[PhraseGroup]) -> int:
        value = 0
        for group in groups:
            for phrase in group.phrases:
                value = max(value, phrase.num_words)
        return value


@total_ordering 
class Reason(tuple[int, str], Enum):
    """
    Describes the "semantic reason" why a piece of text has been segmented (at the end of the text).
    
    Value comparisons can be made between members directly or using the level property
    (eg, "reason1 < reason2" or "reason1.level < reason2.level")

    Configurable side effects associated with reasons, such as pause durations,
    are defined separately.
    """
    
    # For back-compat and as fallback
    UNDEFINED = 0, "undefined"
    # The string has been split after an arbitrary word
    WORD = 1, "w"
    # The string ends at a close-quote and a lowercase continuation follows
    # (ie, an attribution like: "Hello," she said.) - almost no pause
    PHRASE_QUOTE_END = 2, "isqe"
    # The string has been split at a phrase (not to be confused with the class named "Phrase")
    PHRASE = 3, "is" # "is" - think "intra-sentence"
    # The string has been split at a sentence (which does not end in a paragraph break)
    SENTENCE = 4, "s"
    # The string has been split at a paragraph break (line feed)
    PARAGRAPH = 5, "p"
    # The string has been split at a paragraph break *plus* one or more blank lines.
    # Related publishing/typography terms for what this represents: "space break"; "scene break"; "dinkus"
    SPACE_BREAK = 6, "x"
    # The segment was the last text at the end of a section (eg, the end of an html section of an epub). 
    # Think "page break" almost.
    SECTION_BREAK = 7, "xx"

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

    @classmethod
    def from_json_value(cls, value: str | None) -> Reason:
        if value is None:
            return Reason.UNDEFINED
        for member in cls:
            if member.json_value == value:
                return member
        return Reason.UNDEFINED
