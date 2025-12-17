import re
import string

from tts_audiobook_tool.dictionary_en import DictionaryEn

class TextUtil:
    """
    Lower level functions for parsing, processing source text, etc
    """

    _whitesspace_punctuation:set[str] = set()

    @staticmethod
    def is_ws_punc(s: str) -> bool:
        """
        Is the string all punctuation and/or whitespace
        TODO: Am using this for both what it says and also to answer question, "Is string 'vocalizable'?" Untangle.
        """
        if not s:
            return True                
        if not TextUtil._whitesspace_punctuation:
            TextUtil._whitesspace_punctuation = set(
                string.whitespace + string.punctuation + 
                "\u2026\u2013\u2014" + # ellipsis, em-dash, en-dash
                "\u201C\u201D\u2018" # fancy open/closed doublequote, fancy open singlequote
            )
        return all(char in TextUtil._whitesspace_punctuation for char in s)

    @staticmethod
    def massage_for_display(s: str) -> str:
        """ Massage text so it is fit to display in the console on a single line """
        return s.strip() # doesn't do much ATM but yea

    @staticmethod
    def sanitize_for_filename(filename: str) -> str:
        """
        Replaces all non-alpha-numeric characters with underscores,
        replaces consecutive underscores with a single underscore,
        and removes leading/trailing underscores.

        TODO: update using similar logic as elsewhere
            eg: CHINA MIÉVILLE -> CHINA_MI_VILLE.flac

        """
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        collapsed = re.sub(r'_+', '_', sanitized)
        stripped = collapsed.strip('_')
        return stripped

    @staticmethod
    def get_words(s: str, filtered: bool=False) -> list[str]:
        """
        Splits string into words, preserving whitespace
        """

        # Like str.split() but preserves the whitespace at end of each item
        words = re.findall(r'\S+\s*|\s+', s.lstrip())

        if filtered:
            return [word for word in words if not TextUtil.is_ws_punc(word)]
        else:
            return words

    @staticmethod
    def get_words_merged(s: str) -> list[str]:
        """
        Merges "non-content" words with predecessor
        TODO: May not be useful
        """

        words = TextUtil.get_words(s)
        if len(words) == 0:
            return []

        new_words: list[str] = []
        for word in words:
            if new_words and TextUtil.is_ws_punc(word):
                new_words[-1] += word
            else:
                new_words.append(word)

        # Edge case
        if len(new_words) > 1 and TextUtil.is_ws_punc(new_words[0]):
            combined_word = new_words[0] + new_words[1]
            new_words = [combined_word] + new_words[2:]

        return new_words

    @staticmethod
    def get_word_count(s: str, filtered: bool=False) -> int:
        return len( TextUtil.get_words(s, filtered) )

    @staticmethod
    def num_trailing_line_breaks(s: str) -> int:
        trailing_whitespace = s[len(s.rstrip()):]
        return trailing_whitespace.count("\n")

    @staticmethod
    def un_all_caps_english(text: str) -> str:
        """
        If a sentence is all caps, force lowercase on all dictionary words.
        For use with TTS text prompt.
        Generally trying to avoid lines like: "PROLOGUE", which some models can get wrong b/c all-caps.
        """

        has_lower = bool(re.search(r'[a-z]', text))
        if has_lower:
            return text
        has_upper = bool(re.search(r'[A-Z]', text))
        if not has_upper:
            return text

        # String is all caps at this point
        words = text.split(" ")
        new_words = []
        for word in words:
            word = word.strip()
            word_lower = word.lower()
            if DictionaryEn.has(word_lower):
                # It's a recognized english word, so use lowercase
                new_words.append(word_lower)
            else:
                # Don't touch (assumption is that it is or may be an acronym or smth)
                new_words.append(word)

        result = " ".join(new_words)
        return result
