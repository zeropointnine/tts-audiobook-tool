import re
from tts_audiobook_tool.words_dict import Dictionary

class TextUtil:
    """
    App util functions for parsing or transforming source text.
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
    def split_text_into_paragraphs(text: str) -> list[str]:
        """
        Splits a long block of text by the line feed character in the following manner:

        - A new item starts right after a line feed character.
        - Items can have trailing whitespace.
        - Items consisting only of whitespace are merged with the previous item.
        - The total number of characters in the output list is the same as the input string.
        """

        if not text:
            return []

        # Handle the case where the entire text is just whitespace
        if text.isspace():
            return [text]

        # The positive lookbehind `(?<=\n)` splits the string after each newline,
        # keeping the newline character with the preceding item.
        initial_split = re.split(r'(?<=\n)', text)

        # If the text ends with a newline, re.split creates a trailing empty string.
        # This needs to be removed to maintain character count and correctness.
        if initial_split and not initial_split[-1]:
            initial_split.pop()

        # Merge items that consist only of whitespace with the previous item.
        # This satisfies the requirement to not have whitespace-only items,
        # while also preserving the total character count.
        result = []
        for item in initial_split:
            if result and item.isspace():
                result[-1] += item
            else:
                result.append(item)

        return result

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
