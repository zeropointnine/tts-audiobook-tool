import re
import unicodedata
import string

from tts_audiobook_tool.words_dict import Dictionary

class TextUtil:
    """
    Lower level functions for parsing, processing source text, etc
    """

    @staticmethod
    def massage_for_inference(text: str, num2words_lang: str="") -> str:
        """
        Massages text for inference.
        These operations should be common to (ie, compatible with) any tts model.

        TODO: Strip out 'non-verbal' characters, or at least emojis. Use re backslash-w, etc.
        TODO: Strip out stray punctuation-only "words" which do not seem "caesura-related"
        """

        text = text.strip()

        # Replace fancy double-quotes (important for Higgs, eg)
        text = text.replace("“", "\"")
        text = text.replace("”", "\"")

        # Collapse consecutive ellipsis chars
        text = re.sub(r'…+', '…', text)

        # Collapse consecutive dots to triple-dot
        text = re.sub(r'\.{4,}', '...', text)

        # Expand "int words" to prevent TTS model from "naively" verbalizing a string of digits
        if num2words_lang:
            MAX_VALUE = 999
            try:
                from num2words import num2words
                text = re.sub(
                        r'\d+', 
                        lambda x: x.group() if int(x.group()) > MAX_VALUE else num2words(int(x.group()), lang=num2words_lang), 
                        text
                    )        
            except NotImplementedError:
                # Fail silently
                pass

        return text


    @staticmethod
    def is_ws_punc(s: str) -> bool:
        """
        Is the string all punctuation and/or whitespace
        TODO: Ideally we want a test which answers the question, "Is this string 'vocalizable'?"
        """
        if not s:
            return True
        punc_and_white = set(string.whitespace + string.punctuation)
        return all(char in punc_and_white for char in s)

    @staticmethod
    def massage_for_display(s: str) -> str:
        """ Massage text so it is fit to display in the console on a single line """
        return s.strip()

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
    def massage_for_text_comparison(s: str) -> str: 
        """
        Massages source and trancription text can they can be more reliably compared against each other.
        Handles international characters via Unicode-aware regex (backslash-w).
        """
        # Normalize Unicode (fixes 'é' vs 'e'+'´' mismatches)
        s = unicodedata.normalize('NFKC', s)

        # Use casefold for aggressive lowercase (better for Intl text)
        s = s.casefold().strip()

        # Replace fancy apost with normal apost
        s = s.replace("’", "'")

        # Replace non-word characters. 
        # \w matches any Unicode letter or number (and underscore).
        # Logic: Replace if NOT a word char/space/apostrophe OR if it is a standalone apostrophe OR underscore.
        s = re.sub(r"[^\w\s']|(?<!\w)'|'(?!\w)|_", ' ', s)

        # Strip white space
        s = re.sub(r'\s+', ' ', s).strip()

        # TODO: Standardize the text representation of small numbers
        #       Add param num2words_language_code; consider going from spelled-out numbers to digits
        #       Consider value threshold (999? 99?)

        return s

    @staticmethod
    def un_all_caps(text: str) -> str:
        """
        If a sentence is all caps, force lowercase on all dictionary words.
        Generally trying to avoid lines like: "PROLOGUE", which some models will get wrong b/c all-caps
        """

        # String has uppercase letters and no lowercase letters at this point
        has_lower = bool(re.search(r'[a-z]', text))
        if has_lower:
            return text

        has_upper = bool(re.search(r'[A-Z]', text))
        if not has_upper:
            return text

        words = text.split(" ")
        new_words = []
        for word in words:
            word = word.strip()
            word_lower = word.lower()
            if word_lower in Dictionary.words:
                # It's a recognized english word, so use lowercase
                new_words.append(word_lower)
            else:
                # Don't touch (assumption is that it is or may be an acronym or smth)
                new_words.append(word)

        result = " ".join(new_words)
        return result
