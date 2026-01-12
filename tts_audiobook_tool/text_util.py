import re
import string

from tts_audiobook_tool.dictionary_en import DictionaryEn

class TextUtil:
    """
    Lower level functions for parsing, processing source text, etc
    """

    ws_punc_chars: set[str] = set(
        string.whitespace + string.punctuation + 
        "\u2026\u2013\u2014" + # ellipsis, em-dash, en-dash
        "\u201C\u201D\u2018\u2019" # open/closed fancy doublequote, open/closed fancy singlequote
    )

    @staticmethod
    def is_ws_punc(s: str) -> bool:
        """
        Is the string all punctuation and/or whitespace
        TODO: Am using this for both what it says and also to answer question, "Is string not 'vocalizable'?" Untangle.
        """
        if not s:
            return True # TODO: ?...
        return all(char in TextUtil.ws_punc_chars for char in s)

    @staticmethod
    def split_raw_word(raw_word: str) -> tuple[str, str, str]: 
        """
        Separates word into three parts.
        First and last item are consecutive "ws_punc_chars", if any.
        Middle item should be everything in-between.

        param raw_word:
            A single "word", which may have trailing and leading whitespace and/or punctuation

        # TODO: apply this consistently; some places may use similar but incomplete/incorrect logic
        """
        if not raw_word:
            return ("", "", "")
            
        # Find the end of leading ws_punc_chars
        start_idx = 0
        while start_idx < len(raw_word) and raw_word[start_idx] in TextUtil.ws_punc_chars:
            start_idx += 1
            
        # Find the start of trailing ws_punc_chars
        end_idx = len(raw_word) - 1
        while end_idx >= start_idx and raw_word[end_idx] in TextUtil.ws_punc_chars:
            end_idx -= 1
            
        # Extract the three parts
        leading = raw_word[:start_idx]
        middle = raw_word[start_idx:end_idx+1] if start_idx <= end_idx else ""
        trailing = raw_word[end_idx+1:] if end_idx < len(raw_word) - 1 else ""
        
        return (leading, middle, trailing)

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
        Splits string into words, *preserving whitespace*, 
        and optionally filtering out "punctuation words"
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
    
    @staticmethod
    def get_uncommon_words_en(raw_words: list[str]) -> list[tuple[str, int, list[str]]]:
        """
        Returns list of "uncommon" words in descending order of frequency
        Eg: 
            [ 
                ("yggdrasil", 50, ["Yggdrasil", "yggdrasil", "YggDrasil"]), 
                ("zymurgy", 3, ["zymurgy"])
            ]
        """
        counts_dict: dict[str, tuple[int, list[str]]] = {}
        for raw_word in raw_words:
            word = TextUtil.split_raw_word(raw_word)[1]
            word = word.replace("’", "'") # fancy apost
            if not word:
                continue
            word_lc = word.lower() # this is the dict key
            if DictionaryEn.has(word_lc):
                continue
            if word_lc in counts_dict:
                count, instances = counts_dict[word_lc]
            else:
                count = 0
                instances = []
            count += 1
            if not word in instances:
                instances.append(word)
            counts_dict[word_lc] = (count, instances)
        
        sorted_tuples = sorted(
            [(word_lc, count, instances) for word_lc, (count, instances) in counts_dict.items()], key=lambda x: x[1], reverse=True
        )
        return sorted_tuples
