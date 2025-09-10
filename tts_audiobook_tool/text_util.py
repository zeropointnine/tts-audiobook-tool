import re
import string

from tts_audiobook_tool.words_dict import Dictionary

class TextUtil:
    """
    App util functions for parsing or transforming source text.
    """

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
    def expand_int_words_in_text(text: str) -> str:
        """
        Expands any "int words" (from 0 to 999) with spelled-out numbers.
        Eg, "Repeat 28 times." -> "Repeat twenty-eight times."
        """

        # This regex splits the text by whitespace, keeping the whitespace delimiters.
        tokens = re.split(r'(\s+)', text)

        expanded_tokens = []

        # Create a regex pattern for leading and trailing punctuation.
        punct_chars = re.escape(string.punctuation)
        leading_punct_re = re.compile(r'^([' + punct_chars + r']*)')
        trailing_punct_re = re.compile(r'([' + punct_chars + r']*)$')

        for token in tokens:
            if not token:
                continue

            # Pass through whitespace tokens.
            if token.isspace():
                expanded_tokens.append(token)
                continue

            # Find leading punctuation.
            leading_match = leading_punct_re.match(token)
            leading_punct = leading_match.group(1) if leading_match else ''

            # Find trailing punctuation.
            trailing_match = trailing_punct_re.search(token)
            trailing_punct = trailing_match.group(1) if trailing_match else ''

            # Extract the core word.
            start_index = len(leading_punct)
            end_index = len(token) - len(trailing_punct)

            # Handle cases where token is only punctuation or where matches overlap.
            if start_index >= end_index:
                expanded_tokens.append(token)
                continue

            core_word = token[start_index:end_index]

            # Expand the core word using the other static method.
            expanded_core = TextUtil._expand_int_word_or_pass_through(core_word)

            # Reconstruct the token with original punctuation.
            new_token = leading_punct + expanded_core + trailing_punct
            expanded_tokens.append(new_token)

        return "".join(expanded_tokens)

    @staticmethod
    def _expand_int_word_or_pass_through(word: str) -> str:

        word = word.strip()
        is_strict_int = word.isdigit() and (word == '0' or word[0] != '0') # digits only, no leading zeros before other numbers
        if not is_strict_int:
            return word

        n = int(word)
        if not (0 <= n <= 999):
            return word

        if n == 0:
            return "zero"

        # Define the building blocks for the number words
        LESS_THAN_20 = [
            "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        ]
        TENS = [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        ]

        parts = []

        # Step 1: Handle the hundreds place
        if n >= 100:
            # e.g., for 999, this gets the '9' from 999 // 100
            hundreds_digit = n // 100
            parts.append(f"{LESS_THAN_20[hundreds_digit]} hundred")
            # Get the remainder, e.g., 999 % 100 = 99
            n %= 100

        # Step 2: Handle the tens and ones place
        if n >= 20:
            # e.g., for 99, this gets the 'ninety' from TENS[99 // 10]
            parts.append(TENS[n // 10])
            # Get the remainder, e.g., 99 % 10 = 9
            n %= 10
            # If there's a one's digit, add it with a hyphen
            if n > 0:
                parts[-1] += f"-{LESS_THAN_20[n]}"
        elif n > 0:
            # This handles numbers from 1 to 19
            parts.append(LESS_THAN_20[n])

        # Step 3: Join the parts with spaces
        return " ".join(parts)

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
